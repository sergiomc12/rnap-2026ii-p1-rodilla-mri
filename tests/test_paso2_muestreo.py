import pytest

from pregunta1.paso2_preprocesamiento.muestreo import (
    LABELS, EstudioEtiquetado, MuestreadorEstratificado,
)


def estudio(uid, positivos=(), evidencias=5, dorado=False, idioma="en",
            veredicto=None):
    if veredicto is None:
        veredicto = "con_hallazgo" if positivos else "sano_negado"
    return EstudioEtiquetado(
        uid=uid, idioma=idioma,
        etiquetas={e: int(e in positivos) for e in LABELS},
        n_evidencias=evidencias, es_dorado=dorado, veredicto=veredicto,
    )


def test_se_distingue_la_rodilla_sana_del_informe_ilegible():
    """Las dos dan doce ceros, pero solo una sirve para entrenar.

    Confundirlas costaba los ejemplos negativos: la muestra quedaba con un 6%
    de rodillas sanas cuando en el corpus son el 15%.
    """
    assert estudio("a", veredicto="ilegible").ilegible is True
    assert estudio("a", veredicto="ilegible").sano_verificado is False

    for bueno in ("sano_negado", "sano_afirmado"):
        assert estudio("b", veredicto=bueno).ilegible is False
        assert estudio("b", veredicto=bueno).sano_verificado is True

    assert estudio("c", ("ACL",)).ilegible is False
    assert estudio("c", ("ACL",)).sano_verificado is False


def test_todos_los_dorados_entran_pase_lo_que_pase():
    estudios = [estudio(f"oro{i}", dorado=True, evidencias=0) for i in range(5)]
    estudios += [estudio(f"x{i}") for i in range(50)]

    muestra = MuestreadorEstratificado(objetivo=10).muestrear(estudios)

    assert muestra.dorados == 5
    assert all(f"oro{i}" in muestra.uids for i in range(5))


def test_los_ilegibles_se_cuentan_aparte_de_los_sanos():
    """Son dos motivos distintos de descarte y el conteo los distingue.

    El de ilegibles mide cuanto informe no se pudo leer; el de sanos, una
    decision de composicion que se midio (ver el test siguiente).
    """
    estudios = [estudio(f"i{i}", veredicto="ilegible") for i in range(20)]
    estudios += [estudio(f"s{i}", veredicto="sano_negado") for i in range(7)]
    estudios += [estudio(f"u{i}", ("ACL",)) for i in range(5)]

    muestra = MuestreadorEstratificado(objetivo=25).muestrear(estudios)

    assert muestra.descartados_ilegibles == 20
    assert len(muestra.estudios) == 5
    assert muestra.sanos_verificados == 0


def test_se_pueden_incluir_los_ilegibles_a_proposito():
    """Para medir cuanto ruido metian, no para usarlo en produccion."""
    estudios = [estudio(f"i{i}", veredicto="ilegible") for i in range(20)]

    muestra = MuestreadorEstratificado(
        objetivo=10, excluir_ilegibles=False
    ).muestrear(estudios)

    assert len(muestra.estudios) == 10


def test_la_etiqueta_rara_queda_representada_pese_a_ser_minoria():
    comunes = [estudio(f"c{i}", ("Effusion",)) for i in range(200)]
    raros = [estudio(f"r{i}", ("Fracture",)) for i in range(5)]

    muestra = MuestreadorEstratificado(objetivo=50, cuota_por_etiqueta=5).muestrear(
        comunes + raros
    )

    assert muestra.prevalencia()["Fracture"] == 5


def test_no_se_pasa_del_objetivo():
    estudios = [estudio(f"x{i}", ("ACL", "Effusion")) for i in range(500)]

    muestra = MuestreadorEstratificado(objetivo=30).muestrear(estudios)

    assert len(muestra.estudios) == 30


def test_la_misma_semilla_da_la_misma_muestra():
    estudios = [estudio(f"x{i}", ("ACL",)) for i in range(100)]

    primera = MuestreadorEstratificado(objetivo=20, semilla=7).muestrear(estudios)
    segunda = MuestreadorEstratificado(objetivo=20, semilla=7).muestrear(estudios)

    assert set(primera.uids) == set(segunda.uids)


def test_semillas_distintas_dan_muestras_distintas():
    estudios = [estudio(f"x{i}", ("ACL",)) for i in range(200)]

    una = MuestreadorEstratificado(objetivo=20, semilla=1).muestrear(estudios)
    otra = MuestreadorEstratificado(objetivo=20, semilla=2).muestrear(estudios)

    assert set(una.uids) != set(otra.uids)


def test_la_prevalencia_cuenta_las_doce_etiquetas():
    muestra = MuestreadorEstratificado(objetivo=1).muestrear([estudio("a", ("ACL",))])

    assert set(muestra.prevalencia()) == set(LABELS)


@pytest.mark.parametrize("campo,valor", [("objetivo", 0), ("cuota_por_etiqueta", 0)])
def test_el_muestreador_valida_sus_rangos(campo, valor):
    with pytest.raises(ValueError):
        MuestreadorEstratificado(**{campo: valor})


def test_los_sanos_verificados_quedan_fuera_por_defecto():
    """Medido: incluirlos empeora el AUC sobre los dorados en 0.045.

    Un estudio "sano" lo es porque el informe dice que ALGUNA estructura esta
    normal, no porque no haya nada. Entra con doce ceros y la mayoria de esas
    celdas nunca se menciono en el informe: son desconocidas, no negativas.
    """
    estudios = [estudio(f"s{i}", veredicto="sano_negado") for i in range(10)]
    estudios += [estudio(f"a{i}", veredicto="sano_afirmado") for i in range(10)]
    estudios += [estudio(f"u{i}", ("ACL",)) for i in range(5)]

    muestra = MuestreadorEstratificado(objetivo=30).muestrear(estudios)

    assert len(muestra.estudios) == 5
    assert muestra.sanos_verificados == 0


def test_se_pueden_incluir_los_sanos_para_reproducir_la_ablacion():
    estudios = [estudio(f"s{i}", veredicto="sano_negado") for i in range(10)]
    estudios += [estudio(f"u{i}", ("ACL",)) for i in range(5)]

    muestra = MuestreadorEstratificado(
        objetivo=30, incluir_sanos=True
    ).muestrear(estudios)

    assert len(muestra.estudios) == 15
    assert muestra.sanos_verificados == 10
