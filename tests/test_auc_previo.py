"""Reanudar no debe pisar un checkpoint mejor con uno peor.

Existe por una perdida real y silenciosa. La GPU cuelga el entrenamiento cada
una o dos horas --un kernel CUDA que no completa; el proceso queda girando en
la sincronizacion de `float(perdida.detach())`-- y la unica salida practica es
matar y reanudar desde el ultimo checkpoint.

Pero el `Historial` de la corrida reanudada arrancaba con su mejor en cero, asi
que su PRIMERA epoca siempre contaba como mejora y sobrescribia los pesos.
Reanudando desde la epoca 38 con AUC 0.7245, una epoca 39 de 0.7176 borraba
los pesos buenos y `progreso.json` pasaba a decir "mejor epoca 39": nada
indicaba que antes hubo algo superior.
"""

from __future__ import annotations

import pytest

from pregunta1.paso3_modelos.entrenamiento import (  # noqa: E402
    ConfigEntrenamiento, Historial, ResultadoEpoca,
)
from pregunta1.paso4_evaluacion.metricas import LABELS, ResultadoAUC  # noqa: E402


def _epoca(numero: int, auc: float) -> ResultadoEpoca:
    return ResultadoEpoca(numero, 0.4, 0.4,
                          ResultadoAUC({e: auc for e in LABELS}), 1.0)


class TestNoPisaUnCheckpointMejor:
    def test_una_epoca_peor_no_cuenta_como_mejora(self):
        """El caso medido: reanudar en 0.7245 y que la 39 de 0.7176."""
        h = Historial(mejor_auc=0.7245, mejor_epoca=38)
        assert h.registrar(_epoca(39, 0.7176)) is False
        assert h.mejor_epoca == 38, "la mejor epoca no debe moverse"
        assert h.mejor_auc == pytest.approx(0.7245)

    def test_una_epoca_mejor_si_cuenta(self):
        h = Historial(mejor_auc=0.7245, mejor_epoca=38)
        assert h.registrar(_epoca(39, 0.7300)) is True
        assert h.mejor_epoca == 39
        assert h.mejor_auc == pytest.approx(0.7300)

    def test_sin_semilla_la_primera_epoca_siempre_es_la_mejor(self):
        """El comportamiento viejo, que sigue siendo correcto desde cero."""
        h = Historial()
        assert h.registrar(_epoca(1, 0.4)) is True


class TestPaciencia:
    def test_reanudar_no_agota_la_paciencia_de_entrada(self):
        """El error que `epoca_previa` evita.

        Contar `len(epocas) - mejor_epoca` con la numeracion absoluta daria
        negativo o saltaria, y la corrida podria cortar antes de empezar.
        """
        h = Historial(mejor_auc=0.7245, mejor_epoca=38)
        assert h.epocas_sin_mejorar == 0, "sin epocas corridas, no hay espera"

    def test_cuenta_las_epocas_posteriores_a_la_mejor(self):
        h = Historial(mejor_auc=0.7245, mejor_epoca=38)
        for n, auc in [(39, 0.70), (40, 0.71), (41, 0.72)]:
            h.registrar(_epoca(n, auc))
        assert h.epocas_sin_mejorar == 3

    def test_una_mejora_reinicia_la_cuenta(self):
        h = Historial(mejor_auc=0.7245, mejor_epoca=38)
        h.registrar(_epoca(39, 0.70))
        h.registrar(_epoca(40, 0.73))
        assert h.mejor_epoca == 40
        assert h.epocas_sin_mejorar == 0

    def test_desde_cero_sigue_contando_igual(self):
        h = Historial()
        h.registrar(_epoca(1, 0.6))
        h.registrar(_epoca(2, 0.5))
        h.registrar(_epoca(3, 0.5))
        assert h.mejor_epoca == 1
        assert h.epocas_sin_mejorar == 2


class TestConfiguracion:
    def test_por_defecto_no_hay_previo(self):
        cfg = ConfigEntrenamiento()
        assert cfg.auc_previo == 0.0
        assert cfg.epoca_previa == 0

    def test_acepta_el_previo(self):
        cfg = ConfigEntrenamiento(auc_previo=0.7245, epoca_previa=38)
        assert cfg.auc_previo == pytest.approx(0.7245)
        assert cfg.epoca_previa == 38
