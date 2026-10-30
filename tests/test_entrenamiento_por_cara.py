"""Entrenar una cara con solo algunas etiquetas.

Medido a 5 epocas con la misma arquitectura sobre los tres planos, cada cara
es la mejor para un subconjunto distinto: `Synovitis` y `ACL` en axial,
`PF OA` y `Lateral Meniscus` en coronal, las ocho restantes en sagital. Para
entrenar asi hacen falta dos cosas, y las dos se rompen en silencio:

  1. La perdida tiene que promediar sobre las columnas ACTIVAS. Enmascarar
     con ceros y dejar `reduction="mean"` divide por doce, asi que con dos
     activas el gradiente sale seis veces mas chico sin que nada avise.
  2. La mejor epoca tiene que elegirse por el macro de las activas. Elegirla
     por el de las doce es elegir por el AUC de diez cabezas cuyo gradiente
     esta apagado: ruido del backbone compartido.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pregunta1.paso3_modelos.perdidas import (  # noqa: E402
    PerdidaEnmascarada, construir_perdida,
)
from pregunta1.paso4_evaluacion.metricas import LABELS, ResultadoAUC  # noqa: E402


@pytest.fixture
def lote():
    torch.manual_seed(0)
    return torch.randn(4, 12), (torch.rand(4, 12) > 0.5).float()


class TestPerdidaEnmascarada:
    def test_promedia_solo_las_activas(self, lote):
        import torch.nn.functional as F
        logits, objetivo = lote
        perdida = construir_perdida("bce", indices_activos=[0, 8])
        a_mano = F.binary_cross_entropy_with_logits(
            logits, objetivo, reduction="none")[:, [0, 8]].mean()
        assert torch.allclose(perdida(logits, objetivo), a_mano, atol=1e-6)

    def test_no_divide_por_doce(self, lote):
        """El error que esto evita: el gradiente escalado por activas/12."""
        logits, objetivo = lote
        logits = logits.clone().requires_grad_(True)
        construir_perdida("bce", indices_activos=[0, 8])(logits, objetivo).backward()
        correcto = logits.grad.clone()

        logits2 = lote[0].clone().requires_grad_(True)
        mascara = torch.zeros(12)
        mascara[[0, 8]] = 1.0
        import torch.nn as nn
        malo = (nn.BCEWithLogitsLoss(reduction="none")(logits2, objetivo)
                * mascara).mean()
        malo.backward()

        assert not torch.allclose(correcto, logits2.grad, atol=1e-6)
        # El factor es exactamente 12/2 = 6.
        assert torch.allclose(correcto, logits2.grad * 6, atol=1e-5)

    def test_las_inactivas_no_aportan_gradiente(self, lote):
        logits, objetivo = lote
        logits = logits.clone().requires_grad_(True)
        construir_perdida("bce", indices_activos=[8])(logits, objetivo).backward()
        inactivas = [i for i in range(12) if i != 8]
        assert torch.all(logits.grad[:, inactivas] == 0)
        assert torch.any(logits.grad[:, 8] != 0)

    def test_sin_activas_es_la_perdida_normal(self, lote):
        logits, objetivo = lote
        normal = construir_perdida("bce")
        assert not isinstance(normal, PerdidaEnmascarada)
        assert torch.allclose(normal(logits, objetivo),
                              torch.nn.BCEWithLogitsLoss()(logits, objetivo))

    def test_rechaza_lista_vacia(self):
        with pytest.raises(ValueError, match="al menos una"):
            PerdidaEnmascarada(torch.nn.BCEWithLogitsLoss(reduction="none"), [])

    @pytest.mark.parametrize("nombre", ["bce", "bce_ponderada", "focal"])
    def test_las_tres_perdidas_se_pueden_enmascarar(self, lote, nombre):
        logits, objetivo = lote
        perdida = construir_perdida(nombre, [0.1] * 12, indices_activos=[0, 8])
        valor = perdida(logits, objetivo)
        assert valor.ndim == 0 and torch.isfinite(valor)

    def test_el_buffer_viaja_con_el_modulo(self, lote):
        """`indices` como buffer y no tensor suelto: si no, indexa en CPU."""
        perdida = construir_perdida("bce", indices_activos=[0, 8])
        assert "indices" in dict(perdida.named_buffers())


class TestSeleccionDeMejorEpoca:
    def test_macro_de_un_subconjunto(self):
        r = ResultadoAUC(por_etiqueta={e: 0.5 for e in LABELS}
                         | {"Synovitis": 0.9, "ACL": 0.7})
        assert r.macro_de(["Synovitis", "ACL"]) == pytest.approx(0.8)
        assert r.macro < 0.6

    def test_macro_de_ignora_las_no_definidas(self):
        r = ResultadoAUC(por_etiqueta={"ACL": None, "MCL": 0.8})
        assert r.macro_de(["ACL", "MCL"]) == pytest.approx(0.8)

    def test_macro_de_sin_nada_definido_es_cero(self):
        r = ResultadoAUC(por_etiqueta={"ACL": None})
        assert r.macro_de(["ACL"]) == 0.0

    def test_historial_elige_por_las_activas(self):
        """Dos epocas: la 1 mejor en las doce, la 2 mejor en las activas."""
        from pregunta1.paso3_modelos.entrenamiento import Historial, ResultadoEpoca

        base = {e: 0.9 for e in LABELS}
        e1 = ResultadoEpoca(1, 0.4, 0.4, ResultadoAUC(base | {"ACL": 0.50}), 1.0)
        e2 = ResultadoEpoca(2, 0.4, 0.4, ResultadoAUC({e: 0.1 for e in LABELS}
                                                      | {"ACL": 0.99}), 1.0)
        h = Historial(etiquetas_activas=("ACL",))
        assert h.registrar(e1) is True
        assert h.registrar(e2) is True, "la epoca 2 gana en ACL, que es la activa"
        assert h.mejor_epoca == 2
        assert h.mejor_auc == pytest.approx(0.99)

    def test_sin_activas_elige_por_las_doce(self):
        from pregunta1.paso3_modelos.entrenamiento import Historial, ResultadoEpoca

        base = {e: 0.9 for e in LABELS}
        e1 = ResultadoEpoca(1, 0.4, 0.4, ResultadoAUC(base | {"ACL": 0.50}), 1.0)
        e2 = ResultadoEpoca(2, 0.4, 0.4, ResultadoAUC({e: 0.1 for e in LABELS}
                                                      | {"ACL": 0.99}), 1.0)
        h = Historial()
        assert h.registrar(e1) is True
        assert h.registrar(e2) is False, "en las doce la epoca 2 es mucho peor"
        assert h.mejor_epoca == 1

    def test_config_por_defecto_entrena_las_doce(self):
        from pregunta1.paso3_modelos.entrenamiento import ConfigEntrenamiento
        assert ConfigEntrenamiento().etiquetas_activas == ()


class TestRepartoPorCara:
    """El reparto medido a 5 epocas cubre las doce exactamente una vez."""

    REPARTO = {
        "sagital": ("Effusion", "MCL", "Contusion", "Lateral OA", "Baker's",
                    "Fracture", "Medial Meniscus", "Medial OA"),
        "coronal": ("Lateral Meniscus", "PF OA"),
        "axial": ("Synovitis", "ACL"),
    }

    def test_cubre_las_doce_sin_repetir(self):
        todas = [e for grupo in self.REPARTO.values() for e in grupo]
        assert len(todas) == 12
        assert sorted(todas) == sorted(LABELS)

    @pytest.mark.parametrize("cara", ["sagital", "coronal", "axial"])
    def test_los_nombres_existen(self, cara):
        for etiqueta in self.REPARTO[cara]:
            assert etiqueta in LABELS
