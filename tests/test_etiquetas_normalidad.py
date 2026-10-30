"""Pruebas de la distinción entre rodilla sana e informe ilegible.

Las dos producen doce etiquetas en cero, y confundirlas tiene consecuencias
opuestas: tratar una rodilla sana como ilegible quita del entrenamiento el
ejemplo negativo más informativo que hay; tratar un informe ilegible como
rodilla sana le enseña al modelo que un estudio con patología es negativo.
"""

from __future__ import annotations

import pytest

from pregunta1.paso1_etiquetas_debiles.extractor import ExtractorReglas
from pregunta1.paso1_etiquetas_debiles.normalidad import (
    Veredicto, afirma_normalidad, clasificar, es_sano_verificado,
)

EXTRACTOR = ExtractorReglas()


def clasificar_texto(texto: str) -> Veredicto:
    return clasificar(EXTRACTOR.extraer(texto), texto)


class TestAfirmaNormalidad:
    @pytest.mark.parametrize("texto", [
        "Normal medial and lateral menisci.",
        "The cruciate ligaments are intact.",
        "Articular cartilage is unremarkable.",
        "Los meniscos son normales.",
        "Ligamento cruzado anterior íntegro.",
        "No evidence of meniscal tear.",
    ])
    def test_reconoce_la_afirmacion_sobre_una_estructura(self, texto):
        assert afirma_normalidad(texto)

    @pytest.mark.parametrize("texto", [
        "Normal.",
        "The study was performed with a normal protocol.",
        "Examen normal realizado con contraste.",
    ])
    def test_normal_sin_estructura_no_alcanza(self, texto):
        """'Normal' suelto puede ser del protocolo, no de la rodilla."""
        assert not afirma_normalidad(texto)

    def test_exige_la_misma_oracion(self):
        """La estructura en una oración y 'normal' en otra no es una afirmación.

        Sin esta exigencia, cualquier informe que nombre el menisco y en otra
        parte diga que el protocolo fue normal pasaría por rodilla sana.
        """
        texto = "The medial meniscus shows increased signal. Protocol normal."
        assert not afirma_normalidad(texto)

    def test_texto_vacio(self):
        assert not afirma_normalidad("")


class TestClasificar:
    def test_evidencia_toda_negada_es_sano(self):
        """La señal más fuerte: el informe nombra el hallazgo y lo descarta."""
        texto = ("No evidence of knee effusion. No joint effusion is seen.")
        assert clasificar_texto(texto) is Veredicto.NEGADO

    def test_sin_evidencia_pero_afirmando_normalidad(self):
        texto = "The cruciate and collateral ligaments are intact."
        assert clasificar_texto(texto) is Veredicto.AFIRMADO

    def test_sin_evidencia_y_sin_afirmacion_es_ilegible(self):
        """Lo que el extractor no entiende se descarta, no se asume sano."""
        texto = "MRI performed. Sequences acquired. Printed on 2020-01-01."
        assert clasificar_texto(texto) is Veredicto.ILEGIBLE

    def test_un_informe_destrozado_por_la_traduccion_es_ilegible(self):
        texto = "Pour me out. Printed: cross-bands observed. Technique applied."
        assert clasificar_texto(texto) is Veredicto.ILEGIBLE

    def test_la_negacion_pesa_mas_que_la_ausencia_de_frase(self):
        """Con evidencia negada no hace falta que diga 'normal' en ningún lado."""
        texto = "No joint effusion."
        assert clasificar_texto(texto) is Veredicto.NEGADO


class TestEsSanoVerificado:
    def test_un_estudio_con_hallazgo_no_es_sano(self):
        texto = "Tear of the medial meniscus posterior horn."
        assert not es_sano_verificado(EXTRACTOR.extraer(texto), texto)

    def test_un_estudio_negativo_si_lo_es(self):
        texto = "Normal medial and lateral menisci. No joint effusion."
        assert es_sano_verificado(EXTRACTOR.extraer(texto), texto)

    def test_un_estudio_ilegible_no_lo_es(self):
        texto = "Sequences acquired. Printed."
        assert not es_sano_verificado(EXTRACTOR.extraer(texto), texto)

    def test_un_hallazgo_junto_a_una_estructura_normal_no_es_sano(self):
        """Lo habitual en un informe real: algo normal y algo roto."""
        texto = ("The cruciate ligaments are intact. "
                 "There is a tear of the medial meniscus.")
        extraccion = EXTRACTOR.extraer(texto)
        assert "Medial Meniscus" in extraccion.positivos()
        assert not es_sano_verificado(extraccion, texto)


class TestVeredicto:
    def test_los_dos_sanos_cuentan_como_sanos(self):
        assert Veredicto.NEGADO.es_sano
        assert Veredicto.AFIRMADO.es_sano

    def test_ilegible_no(self):
        assert not Veredicto.ILEGIBLE.es_sano
