import numpy as np
import pytest

from pregunta1.paso2_preprocesamiento.geometria import (
    MetadatosCorte, banda_central, es_rodilla_izquierda, normalizar_intensidad,
    ordenar_por_posicion, recortar_milimetros,
)


def meta(espaciado=0.5, posicion=None, lateralidad="", filas=320):
    return MetadatosCorte(filas=filas, columnas=filas, espaciado_mm=espaciado,
                          posicion=posicion, lateralidad=lateralidad)


def test_recorta_la_misma_extension_fisica_pese_a_resoluciones_distintas():
    gruesa = recortar_milimetros(np.zeros((320, 320)), espaciado_mm=0.5, crop_mm=130)
    fina = recortar_milimetros(np.zeros((800, 800)), espaciado_mm=0.2, crop_mm=130)

    assert gruesa.shape == (260, 260)   # 130mm / 0.5
    assert fina.shape == (650, 650)     # 130mm / 0.2


def test_una_adquisicion_mas_chica_que_el_recorte_queda_intacta():
    """Rellenar inventaria anatomia que nunca se escaneo."""
    imagen = np.zeros((200, 200))

    assert recortar_milimetros(imagen, espaciado_mm=1.0, crop_mm=400).shape == (200, 200)


def test_un_espaciado_invalido_no_recorta():
    imagen = np.zeros((320, 320))

    assert recortar_milimetros(imagen, espaciado_mm=0.0).shape == (320, 320)


def test_el_recorte_queda_centrado():
    imagen = np.zeros((100, 100))
    imagen[50, 50] = 1.0

    recortada = recortar_milimetros(imagen, espaciado_mm=1.0, crop_mm=20)

    assert recortada[10, 10] == 1.0


def test_la_intensidad_se_lleva_al_rango_unitario():
    imagen = np.array([[0, 636], [100, 2781]], dtype=np.uint16)

    normalizada = normalizar_intensidad(imagen)

    assert normalizada.min() >= 0.0 and normalizada.max() <= 1.0


def test_dos_escalas_distintas_quedan_comparables():
    """La MRI no tiene unidades absolutas: el mismo tejido lee distinto por escaner."""
    tenue = np.array([[0, 100], [200, 300]], dtype=np.float32)

    assert np.allclose(normalizar_intensidad(tenue), normalizar_intensidad(tenue * 9))


def test_una_imagen_plana_no_divide_por_cero():
    assert normalizar_intensidad(np.full((4, 4), 7.0)).max() == 0.0


def test_la_banda_central_descarta_los_extremos():
    assert banda_central(30, (0.2, 0.8)) == slice(6, 24)


def test_la_banda_nunca_queda_vacia():
    recorte = banda_central(2, (0.2, 0.8))

    assert recorte.stop > recorte.start


def test_la_lateralidad_sale_del_tag_cuando_esta():
    assert es_rodilla_izquierda([meta(lateralidad="L")]) is True
    assert es_rodilla_izquierda([meta(lateralidad="RIGHT")]) is False


def test_sin_tag_se_infiere_del_sistema_de_coordenadas():
    """+x apunta a la izquierda del paciente."""
    assert es_rodilla_izquierda([meta(posicion=(45.0, 0, 0))]) is True
    assert es_rodilla_izquierda([meta(posicion=(-45.0, 0, 0))]) is False


def test_sin_tag_ni_posicion_devuelve_none_en_vez_de_adivinar():
    assert es_rodilla_izquierda([meta()]) is None


def test_los_cortes_se_ordenan_por_posicion_no_por_nombre():
    metadatos = [meta(posicion=(3.0, 0, 0)), meta(posicion=(1.0, 0, 0)),
                 meta(posicion=(2.0, 0, 0))]

    assert ordenar_por_posicion(range(3), metadatos) == [1, 2, 0]


def test_sin_posicion_se_respeta_el_orden_original():
    assert ordenar_por_posicion(range(3), [meta(), meta(), meta()]) == [0, 1, 2]
