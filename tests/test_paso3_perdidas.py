import pytest
import torch
import torch.nn as nn

from pregunta1.paso3_modelos.perdidas import (
    PerdidaFocal, construir_perdida, pesos_por_prevalencia,
)


def test_la_focal_castiga_mucho_menos_un_ejemplo_facil_que_la_bce():
    logits, objetivo = torch.tensor([[3.0, -3.0]]), torch.tensor([[1.0, 0.0]])

    bce = float(nn.BCEWithLogitsLoss()(logits, objetivo))
    focal = float(PerdidaFocal()(logits, objetivo))

    assert focal < bce / 100


def test_la_focal_conserva_el_castigo_de_un_ejemplo_dificil():
    logits, objetivo = torch.tensor([[-3.0, 3.0]]), torch.tensor([[1.0, 0.0]])

    bce = float(nn.BCEWithLogitsLoss()(logits, objetivo))
    focal = float(PerdidaFocal()(logits, objetivo))

    assert focal > bce / 4


def test_gamma_cero_sin_alfa_equivale_a_bce():
    logits, objetivo = torch.randn(4, 12), torch.randint(0, 2, (4, 12)).float()

    focal = PerdidaFocal(gamma=0.0, alpha=None)(logits, objetivo)

    assert torch.allclose(focal, nn.BCEWithLogitsLoss()(logits, objetivo), atol=1e-6)


def test_la_perdida_focal_valida_sus_argumentos():
    with pytest.raises(ValueError):
        PerdidaFocal(gamma=-1.0)
    with pytest.raises(ValueError):
        PerdidaFocal(reduccion="otra")


def test_una_etiqueta_rara_recibe_mas_peso_que_una_comun():
    raro, comun = pesos_por_prevalencia([0.122]), pesos_por_prevalencia([0.507])

    assert float(raro[0]) > float(comun[0])


def test_el_peso_tiene_tope_para_no_desestabilizar():
    assert float(pesos_por_prevalencia([0.001], tope=10.0)[0]) == 10.0


def test_una_prevalencia_cero_no_divide_por_cero():
    assert float(pesos_por_prevalencia([0.0], tope=10.0)[0]) == 10.0


def test_la_fabrica_construye_las_tres_perdidas():
    assert construir_perdida("bce") is not None
    assert construir_perdida("focal") is not None
    assert construir_perdida("bce_ponderada", [0.1] * 12) is not None


def test_la_bce_ponderada_exige_las_prevalencias():
    with pytest.raises(ValueError, match="prevalencias"):
        construir_perdida("bce_ponderada")
