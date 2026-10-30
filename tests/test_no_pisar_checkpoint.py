"""El guardado final no debe destruir un checkpoint mejor.

Existe por una perdida real y silenciosa. Con `auc_previo`, una corrida
reanudada arranca sabiendo cual fue el mejor AUC de la fase anterior. Si
ninguna de sus epocas lo supera:

  * `ajustar` deja `mejores_pesos` en None --no hubo nada mejor que guardar--
    y por lo tanto NO restaura nada: el modelo queda con los pesos de la
    ULTIMA epoca;
  * `_guardar` escribia esos pesos sobre `pesos.pt` al terminar.

Resultado: `pesos.pt` pasaba a contener la ultima epoca mientras
`progreso.json` seguia anunciando la mejor. Paso con resnet50 sagital: el pico
de 0.8055 de la epoca 8 quedo sobrescrito por los pesos de la 60 (0.7588), y
nada en los metadatos lo delataba.
"""

from __future__ import annotations

import pytest

from pregunta1.paso3_modelos.entrenamiento import Historial, ResultadoEpoca
from pregunta1.paso4_evaluacion.metricas import LABELS, ResultadoAUC


def _epoca(numero: int, auc: float) -> ResultadoEpoca:
    return ResultadoEpoca(numero, 0.4, 0.4,
                          ResultadoAUC({e: auc for e in LABELS}), 1.0)


class TestMejoroAqui:
    def test_sin_epocas_no_mejoro(self):
        assert Historial().mejoro_aqui is False

    def test_desde_cero_la_primera_epoca_cuenta(self):
        h = Historial()
        h.registrar(_epoca(1, 0.6))
        assert h.mejoro_aqui is True

    def test_reanudada_que_no_supera_el_piso(self):
        """El caso que destruia el checkpoint: mejor=8, corrio 60-60."""
        h = Historial(mejor_auc=0.8055, mejor_epoca=8)
        h.registrar(_epoca(60, 0.7588))
        assert h.mejor_epoca == 8
        assert h.mejoro_aqui is False, (
            "la epoca 8 no es de esta corrida: pesos.pt NO debe tocarse")

    def test_reanudada_que_si_supera_el_piso(self):
        h = Historial(mejor_auc=0.7245, mejor_epoca=38)
        h.registrar(_epoca(39, 0.7100))
        h.registrar(_epoca(40, 0.7300))
        assert h.mejor_epoca == 40
        assert h.mejoro_aqui is True

    def test_mejora_y_despues_empeora_sigue_contando(self):
        """La mejor sigue siendo de esta corrida aunque termine peor."""
        h = Historial(mejor_auc=0.70, mejor_epoca=10)
        h.registrar(_epoca(11, 0.75))
        h.registrar(_epoca(12, 0.60))
        h.registrar(_epoca(13, 0.58))
        assert h.mejor_epoca == 11
        assert h.mejoro_aqui is True

    @pytest.mark.parametrize("piso,epocas,espera", [
        (0.0, [(1, 0.5)], True),
        (0.9, [(5, 0.5), (6, 0.6)], False),
        (0.5, [(5, 0.4), (6, 0.55)], True),
        (0.5, [(5, 0.50)], False),          # empate no es mejora
    ])
    def test_casos(self, piso, epocas, espera):
        h = Historial(mejor_auc=piso, mejor_epoca=(4 if piso else 0))
        for n, auc in epocas:
            h.registrar(_epoca(n, auc))
        assert h.mejoro_aqui is espera


class TestCoherenciaConProgreso:
    def test_el_mejor_auc_no_baja_nunca(self):
        """Sembrado con 0.8055, una epoca peor no puede bajar el registro."""
        h = Historial(mejor_auc=0.8055, mejor_epoca=8)
        for n, auc in [(60, 0.7588), (61, 0.70), (62, 0.80)]:
            h.registrar(_epoca(n, auc))
        assert h.mejor_auc == pytest.approx(0.8055)
        assert h.mejor_epoca == 8
