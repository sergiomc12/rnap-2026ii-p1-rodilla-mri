import pytest
import torch

from pregunta1.paso3_modelos.pooling import (
    AtencionPooling, PromedioPooling, construir_pooling,
)


def test_la_atencion_reduce_los_grupos_a_un_embedding():
    salida = AtencionPooling(64)(torch.randn(4, 8, 64))

    assert salida.embedding.shape == (4, 64)
    assert salida.atencion.shape == (4, 8)


def test_los_pesos_de_atencion_suman_uno():
    salida = AtencionPooling(64)(torch.randn(4, 8, 64))

    assert torch.allclose(salida.atencion.sum(dim=1), torch.ones(4), atol=1e-5)


def test_los_pesos_de_atencion_no_son_negativos():
    assert (AtencionPooling(32)(torch.randn(2, 6, 32)).atencion >= 0).all()


def test_el_promedio_reparte_el_peso_por_igual():
    salida = PromedioPooling()(torch.randn(3, 5, 16))

    assert torch.allclose(salida.atencion, torch.full((3, 5), 0.2))


def test_el_promedio_es_la_media_de_los_grupos():
    entrada = torch.randn(2, 4, 8)

    assert torch.allclose(PromedioPooling()(entrada).embedding, entrada.mean(dim=1))


def test_la_atencion_sirve_con_cualquier_cantidad_de_grupos():
    """La cantidad de cortes por serie va de 19 a 46 en los datos reales."""
    pooling = AtencionPooling(32)

    assert pooling(torch.randn(1, 3, 32)).embedding.shape == (1, 32)
    assert pooling(torch.randn(1, 20, 32)).embedding.shape == (1, 32)


def test_se_puede_apagar_la_compuerta_para_la_ablacion():
    sin_compuerta = AtencionPooling(32, con_compuerta=False)

    assert sin_compuerta.compuerta is None


@pytest.mark.parametrize("entrada", [torch.randn(4, 64), torch.randn(2, 3, 4, 5)])
def test_una_forma_equivocada_falla_en_vez_de_propagarse(entrada):
    with pytest.raises(ValueError):
        AtencionPooling(64)(entrada)


def test_la_fabrica_construye_las_tres_variantes():
    for nombre in ("atencion", "atencion_sin_compuerta", "promedio"):
        assert construir_pooling(nombre, 32) is not None


def test_un_pooling_desconocido_falla():
    with pytest.raises(ValueError, match="desconocido"):
        construir_pooling("inexistente", 32)
