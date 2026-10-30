import re

import pytest

from pregunta1.paso1_etiquetas_debiles.negacion import EstadoAsercion, clasificar_asercion


def estado(oracion, termino):
    coincidencia = re.search(termino, oracion, re.IGNORECASE)
    return clasificar_asercion(oracion, *coincidencia.span())


@pytest.mark.parametrize("oracion,termino", [
    ("No fracture is seen.", "fracture"),
    ("The knee is without effusion.", "effusion"),
    ("Negative for meniscal tear.", "meniscal"),
    ("Sin rotura meniscal.", "rotura"),
])
def test_un_disparador_previo_niega(oracion, termino):
    assert estado(oracion, termino) is EstadoAsercion.NEGADO


@pytest.mark.parametrize("oracion,termino", [
    ("The ACL is intact.", "ACL"),
    ("Medial meniscus is not torn.", "meniscus"),
    ("The MCL is preserved.", "MCL"),
    ("Ligamentos cruzados dentro de limites normales.", "cruzados"),
    ("Menisco medial de senal conservada.", "menisco medial"),
])
def test_un_disparador_posterior_niega(oracion, termino):
    assert estado(oracion, termino) is EstadoAsercion.NEGADO


@pytest.mark.parametrize("oracion,termino", [
    ("Horizontal tear at the lateral meniscus is noted.", "meniscus"),
    ("Moderate joint effusion is observed.", "effusion"),
    ("Leve derrame articular.", "derrame"),
])
def test_sin_disparadores_queda_afirmado(oracion, termino):
    assert estado(oracion, termino) is EstadoAsercion.AFIRMADO


@pytest.mark.parametrize("oracion,termino", [
    ("Possible tear of the medial meniscus.", "meniscus"),
    ("Findings suspicious for a meniscal tear.", "meniscal"),
    ("Sospecha de rotura meniscal.", "rotura"),
])
def test_las_expresiones_dudosas_marcan_incertidumbre(oracion, termino):
    assert estado(oracion, termino) is EstadoAsercion.INCIERTO


def test_un_disparador_lejano_no_alcanza_al_termino():
    oracion = ("No acute osseous abnormality of the distal femur or proximal "
               "tibia was identified on any sequence, tear of the meniscus")

    assert estado(oracion, "tear") is EstadoAsercion.AFIRMADO


def test_la_ventana_es_configurable():
    oracion = "no evidence of one two three four tear"

    assert clasificar_asercion(oracion, oracion.index("tear"),
                               len(oracion), ventana=2) is EstadoAsercion.AFIRMADO
    assert clasificar_asercion(oracion, oracion.index("tear"),
                               len(oracion), ventana=9) is EstadoAsercion.NEGADO
