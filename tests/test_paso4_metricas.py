import numpy as np
import pytest

from pregunta1.paso4_evaluacion.metricas import (
    auc_macro, auc_roc, bootstrap_auc_macro, mejor_umbral, metricas_en_umbral,
)


def test_una_separacion_perfecta_da_auc_uno():
    assert auc_roc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_el_orden_invertido_da_auc_cero():
    assert auc_roc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_los_empates_dan_azar_en_vez_de_un_valor_arbitrario():
    assert auc_roc([0.5] * 4, [0, 0, 1, 1]) == 0.5


def test_una_etiqueta_con_una_sola_clase_devuelve_none():
    """Devolver 0.5 arrastraria la macro hacia el azar sin avisar."""
    assert auc_roc([0.1, 0.2], [0, 0]) is None
    assert auc_roc([0.1, 0.2], [1, 1]) is None


def test_la_macro_ignora_las_etiquetas_indefinidas():
    puntajes = np.array([[0.9, 0.5], [0.1, 0.5]])
    verdad = np.array([[1, 0], [0, 0]])          # la segunda etiqueta no tiene positivos

    resultado = auc_macro(puntajes, verdad, ("a", "b"))

    assert resultado.por_etiqueta["b"] is None
    assert resultado.n_indefinidas == 1
    assert resultado.macro == 1.0


def test_formas_distintas_fallan_en_vez_de_propagarse():
    with pytest.raises(ValueError):
        auc_macro(np.zeros((4, 3)), np.zeros((4, 2)))


def test_el_bootstrap_devuelve_un_intervalo_que_contiene_al_valor():
    azar = np.random.RandomState(0)
    verdad = azar.randint(0, 2, (56, 12))
    puntajes = verdad * 0.4 + azar.rand(56, 12) * 0.6

    intervalo = bootstrap_auc_macro(puntajes, verdad, remuestreos=200)

    assert intervalo.bajo <= intervalo.valor <= intervalo.alto


def test_el_bootstrap_es_reproducible():
    azar = np.random.RandomState(1)
    verdad, puntajes = azar.randint(0, 2, (40, 12)), azar.rand(40, 12)

    primero = bootstrap_auc_macro(puntajes, verdad, 100, semilla=7)
    segundo = bootstrap_auc_macro(puntajes, verdad, 100, semilla=7)

    assert (primero.bajo, primero.alto) == (segundo.bajo, segundo.alto)


def test_un_umbral_alto_baja_el_recall():
    puntajes = np.array([[0.6], [0.3]])
    verdad = np.array([[1], [1]])

    assert metricas_en_umbral(puntajes, verdad, 0.5).recall == 0.5
    assert metricas_en_umbral(puntajes, verdad, 0.2).recall == 1.0


def test_el_mejor_umbral_maximiza_el_f1():
    puntajes = np.array([[0.9], [0.8], [0.2], [0.1]])
    verdad = np.array([[1], [1], [0], [0]])

    assert mejor_umbral(puntajes, verdad).f1 == 1.0
