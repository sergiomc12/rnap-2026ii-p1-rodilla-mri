"""Assertion status of a finding inside a sentence.

Radiology reports enumerate what the radiologist did NOT find as much as
what they did: "ACL is intact", "no fracture is seen", "medial meniscus is
not torn". A term matcher that ignores this marks positive exactly where
the report says the structure is healthy -- measured at 18 false positives
out of 58 gold studies for ACL alone.

The approach follows NegEx: a window around the term is scanned for
triggers that negate it, and the trigger's position relative to the term
decides whether it applies.
"""

from __future__ import annotations

import re
from enum import Enum

VENTANA_PALABRAS = 6  # how far a trigger reaches, in words


class EstadoAsercion(str, Enum):
    """What the sentence says about a finding it mentions."""

    AFIRMADO = "afirmado"
    NEGADO = "negado"
    INCIERTO = "incierto"


# Triggers that negate what comes AFTER them: "no fracture", "sin rotura".
_PRE_NEGACION = (
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\bfree of\b",
    r"\babsence of\b", r"\babsent\b", r"\bnone\b", r"\bdenies\b",
    r"\bnor\b", r"\brules out\b", r"\bruled out\b",
    r"\bsin\b", r"\bno se\b", r"\bausencia de\b", r"\bausente\b",
)

# Triggers that negate what comes BEFORE them: "ACL is intact",
# "meniscus is not torn", "ligaments within normal limits".
_POST_NEGACION = (
    r"\bis intact\b", r"\bare intact\b", r"\bintact\b",
    r"\bis preserved\b", r"\bare preserved\b", r"\bpreserved\b",
    r"\bis normal\b", r"\bare normal\b", r"\bnormal\b",
    r"\bunremarkable\b", r"\bwithin normal limits\b",
    r"\bis not torn\b", r"\bare not torn\b", r"\bnot torn\b",
    r"\bno evidence\b", r"\bis seen\b(?=\s*$)",
    r"\bintacto\b", r"\bintactos\b", r"\bconservado\b", r"\bconservada\b",
    r"\bnormales\b", r"\bsin alteraciones\b",
    r"\bdentro de l[ií]mites normales\b", r"\bsin signos de\b",
)

# Hedging that makes the finding uncertain rather than affirmed or denied.
_INCERTIDUMBRE = (
    r"\bpossible\b", r"\bprobable\b", r"\bsuspicion\b", r"\bsuspicious\b",
    r"\bcannot be excluded\b", r"\bcannot exclude\b", r"\bequivocal\b",
    r"\bquestionable\b", r"\bmay represent\b", r"\bsuggestive of\b",
    r"\bposible\b", r"\bsospecha\b", r"\bno se puede descartar\b",
    r"\bsugestivo de\b", r"\bdudoso\b",
)


def _compilar(patrones: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(patrones), re.IGNORECASE)


_PRE_RE = _compilar(_PRE_NEGACION)
_POST_RE = _compilar(_POST_NEGACION)
_INCIERTO_RE = _compilar(_INCERTIDUMBRE)


def clasificar_asercion(
    oracion: str, inicio: int, fin: int, ventana: int = VENTANA_PALABRAS,
    solo_previa: bool = False,
) -> EstadoAsercion:
    """Say whether the term at [inicio, fin) is affirmed, negated or hedged.

    `inicio` and `fin` are character offsets of the matched term inside
    `oracion`, as returned by `re.Match.span()`.
    """
    antes = _ultimas_palabras(oracion[:inicio], ventana)
    despues = _primeras_palabras(oracion[fin:], ventana)

    if _PRE_RE.search(antes) or (solo_previa is False and _POST_RE.search(despues)):
        return EstadoAsercion.NEGADO
    if _INCIERTO_RE.search(antes) or _INCIERTO_RE.search(despues):
        return EstadoAsercion.INCIERTO
    return EstadoAsercion.AFIRMADO


def _ultimas_palabras(texto: str, cantidad: int) -> str:
    return " ".join(texto.split()[-cantidad:])


def _primeras_palabras(texto: str, cantidad: int) -> str:
    return " ".join(texto.split()[:cantidad])
