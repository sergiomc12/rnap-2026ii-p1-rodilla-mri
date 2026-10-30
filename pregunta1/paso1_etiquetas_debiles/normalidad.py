"""Distinguir una rodilla sana de un informe que el extractor no supo leer.

Las dos cosas producen el mismo resultado —doce etiquetas en cero— y durante
todo el proyecto se trataron igual: el muestreo descartaba cualquier estudio
sin evidencia, bajo el supuesto, escrito en el código, de que "no son rodillas
sanas, son informes que el extractor no pudo leer".

Medido, ese supuesto es falso en la mayoría de los casos. De los 805 estudios
no dorados que quedan sin ninguna etiqueta, **612 afirman explícitamente que
alguna estructura es normal**. Descartarlos no quitaba ruido: quitaba los
ejemplos negativos, y dejaba el entrenamiento con un 6% de rodillas sanas
cuando en el corpus son el 15%. Encaja con lo que el modelo hace: precisión
0.466 contra recall 0.744, y umbrales óptimos entre 0.10 y 0.30 — sobre-predice
hallazgos.

Un estudio se da por sano cuando el informe lo dice, no cuando callar es
ambiguo. Hay dos formas de decirlo, y se distinguen porque no valen lo mismo:

* ``NEGADO`` — el extractor encontró la estructura mencionada y clasificó
  **toda** la evidencia como negada ("No evidence of knee effusion"). Es la
  señal más fuerte: el informe nombra el hallazgo y lo descarta.
* ``AFIRMADO`` — no hubo evidencia, pero alguna oración afirma que una
  estructura concreta está normal ("Normal medial and lateral menisci"). Más
  débil, porque depende de una frase suelta.

Lo que no entra en ninguna de las dos se marca ``ILEGIBLE`` y se sigue
descartando: ahí sí no sabemos qué dice el informe.
"""

from __future__ import annotations

import re
from enum import Enum

from .negacion import EstadoAsercion
from .oraciones import dividir_en_oraciones


class Veredicto(str, Enum):
    """Por qué un estudio quedó sin ninguna etiqueta positiva."""

    NEGADO = "sano_negado"
    AFIRMADO = "sano_afirmado"
    ILEGIBLE = "ilegible"

    @property
    def es_sano(self) -> bool:
        return self is not Veredicto.ILEGIBLE


def _re(patron: str) -> re.Pattern:
    return re.compile(patron, re.IGNORECASE)


# Afirmaciones de normalidad. "no evidence of" entra acá además de en la
# detección de negación porque una oración puede descartar un hallazgo que el
# léxico no reconoce, y sigue siendo una afirmación de normalidad.
_NORMALIDAD = _re(
    r"\bnormal\b|\bintact\b|\bunremarkable\b|\borderly\b|\bpreserved\b"
    r"|\bno evidence of\b|\bwithin normal limits\b|\bno (?:significant )?"
    r"(?:abnormalit|patholog|lesion|tear|rupture)"
    r"|normales?\b|\bíntegr|\bconservad|sin (?:alteraciones|hallazgos|lesiones)"
    r"|no se (?:observan|identifican|aprecian)"
)

# La afirmación tiene que ser SOBRE una estructura de la rodilla. Sin esto,
# un informe ilegible que contenga la palabra "normal" por casualidad pasaría
# por rodilla sana.
_ESTRUCTURA = _re(
    r"\bmenisc|\bligament|\bcruciate\b|\bcollateral\b|\bcartilage\b"
    r"|\bpatell|\btrochlea|\bfemor|\btibia|\bbone marrow\b|\bjoint\b"
    r"|\beffusion\b|\bsynovi|\btendon\b|\bbursa"
    r"|menisco|ligamento|cruzado|colateral|cart[ií]lago|r[oó]tula"
    r"|f[ée]mur|tibia|m[ée]dula [oó]sea|articula|derrame|sinovi|tend[oó]n"
)


def afirma_normalidad(reporte: str) -> bool:
    """¿Alguna oración dice que una estructura concreta de la rodilla está bien?

    Se exige que las dos cosas caigan en la **misma oración**. "Normal" suelto
    en un informe por lo demás ilegible no alcanza: el criterio es que el texto
    afirme algo, no que contenga una palabra.
    """
    for oracion in dividir_en_oraciones(reporte):
        if _NORMALIDAD.search(oracion) and _ESTRUCTURA.search(oracion):
            return True
    return False


def clasificar(extraccion, reporte: str) -> Veredicto:
    """Por qué este estudio no tiene ninguna etiqueta positiva.

    Solo tiene sentido llamarla sobre estudios sin positivos; con positivos el
    estudio no es un caso dudoso y no hace falta clasificarlo.
    """
    if extraccion.evidencias and all(
        e.estado is EstadoAsercion.NEGADO for e in extraccion.evidencias
    ):
        return Veredicto.NEGADO
    if afirma_normalidad(reporte):
        return Veredicto.AFIRMADO
    return Veredicto.ILEGIBLE


def es_sano_verificado(extraccion, reporte: str) -> bool:
    """Atajo: ¿este estudio se puede usar como ejemplo negativo?"""
    if extraccion.positivos():
        return False
    return clasificar(extraccion, reporte).es_sano
