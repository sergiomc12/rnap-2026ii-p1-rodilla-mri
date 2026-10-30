import numpy as np
import pytest

from pregunta1.paso2_preprocesamiento.geometria import MetadatosCorte
from pregunta1.paso2_preprocesamiento.preprocesamiento import (
    ConfigPreproceso, indices_de_grupos, preprocesar_serie, redimensionar,
)


def serie(n=30, lado=320, espaciado=0.5, lateralidad=""):
    imagenes = [np.random.RandomState(i).randint(0, 1000, (lado, lado)).astype(np.uint16)
                for i in range(n)]
    metadatos = [MetadatosCorte(filas=lado, columnas=lado, espaciado_mm=espaciado,
                                posicion=(float(i), 0.0, 0.0), lateralidad=lateralidad)
                 for i in range(n)]
    return imagenes, metadatos


def test_la_forma_de_salida_es_la_declarada_por_la_config():
    config = ConfigPreproceso()

    tensor = preprocesar_serie(*serie(), config)

    assert tensor.shape == config.forma == (8, 3, 224, 224)


def test_resoluciones_distintas_dan_la_misma_forma():
    chica = preprocesar_serie(*serie(lado=320, espaciado=0.5))
    grande = preprocesar_serie(*serie(lado=800, espaciado=0.225))

    assert chica.shape == grande.shape


def test_la_salida_queda_normalizada():
    tensor = preprocesar_serie(*serie())

    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_una_serie_vacia_falla_en_vez_de_devolver_ceros():
    with pytest.raises(ValueError):
        preprocesar_serie([], [])


def test_una_serie_mas_corta_que_el_grupo_se_rellena_en_vez_de_perderse():
    tensor = preprocesar_serie(*serie(n=2), ConfigPreproceso(grupos=2))

    assert tensor.shape == (2, 3, 224, 224)


def test_la_rodilla_izquierda_invierte_el_orden_de_cortes():
    izquierda = preprocesar_serie(*serie(lateralidad="L"))
    derecha = preprocesar_serie(*serie(lateralidad="R"))

    assert not np.allclose(izquierda, derecha)


def test_se_puede_desactivar_la_normalizacion_de_lateralidad():
    config = ConfigPreproceso(normalizar_lateralidad=False)

    izquierda = preprocesar_serie(*serie(lateralidad="L"), config)
    derecha = preprocesar_serie(*serie(lateralidad="R"), config)

    assert np.allclose(izquierda, derecha)


def test_los_grupos_se_reparten_a_lo_largo_de_la_serie():
    grupos = indices_de_grupos(list(range(20)), grupos=4, por_grupo=3)

    assert [g[0] for g in grupos] == [0, 6, 11, 17]
    assert all(len(g) == 3 for g in grupos)


def test_cada_grupo_son_cortes_contiguos():
    """Lo que hace que sea 2.5D y no tres imagenes sin relacion."""
    for grupo in indices_de_grupos(list(range(20)), grupos=3, por_grupo=3):
        assert grupo == list(range(grupo[0], grupo[0] + 3))


def test_un_solo_grupo_se_toma_del_centro():
    assert indices_de_grupos(list(range(20)), grupos=1, por_grupo=3)[0] == [8, 9, 10]


def test_redimensionar_respeta_el_lado_pedido():
    assert redimensionar(np.zeros((800, 800), np.float32), 224).shape == (224, 224)


def test_redimensionar_no_toca_lo_que_ya_tiene_el_tamano():
    imagen = np.zeros((224, 224), np.float32)

    assert redimensionar(imagen, 224) is imagen


@pytest.mark.parametrize("campo,valor", [
    ("crop_mm", 0), ("lado", 16), ("cortes_por_grupo", 0), ("grupos", 0),
])
def test_la_config_valida_sus_rangos(campo, valor):
    with pytest.raises(ValueError):
        ConfigPreproceso(**{campo: valor})


def test_la_banda_debe_ser_creciente_y_estar_en_cero_uno():
    with pytest.raises(ValueError):
        ConfigPreproceso(banda=(0.8, 0.2))
