"""Splitting radiology reports into sentences.

Reports list one finding per sentence, so the unit of extraction is the
sentence, not the report: working on the whole report lets "tear" from one
sentence bind to "lateral meniscus" from another.

These texts do not split cleanly with a generic splitter: they carry
measurements ("8x12mm."), abbreviations ("approx."), decimal numbers
("1.2x1.5cm.") and placeholders ("[DATE]"), all of which a naive split on
"." breaks apart.

Reports also pad their findings with runs of several spaces. Splitting on
those as well was tried and measured against the 58 gold studies at
thresholds of 2, 3, 4, 6 and 8 spaces: macro agreement stayed at 72.1% and
macro F1 at 0.512 in every case, because those runs always accompany a
period that already ends the sentence. The rule was removed rather than
kept at a threshold chosen arbitrarily.
"""

from __future__ import annotations

import re
from typing import Iterator

# Abbreviations whose trailing period does not end a sentence.
ABREVIATURAS = frozenset({
    "approx", "aprox", "cf", "dr", "etc", "fig", "mm", "cm", "ca",
    "vs", "no", "nr", "ref", "sec", "min", "max", "mg", "ml", "hr",
})

# Candidate terminators: resolved one by one, because a variable-width
# lookbehind (needed for the abbreviation list) is not allowed in `re`.
_CANDIDATO = re.compile(r"[.!?]+(?=\s|$)")

_ULTIMA_PALABRA = re.compile(r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)$")

_MINIMO_CARACTERES = 3

# Separadores "duros" de viñeta: muchos reportes listan un hallazgo por viñeta
# ">" / "•" / "* " SIN cerrar con punto, de modo que el divisor por puntuacion
# los une en una sola "oracion" y una estructura negada en una viñeta se
# empareja con una patologia de otra (medido: "no increased intensity in the
# ACL, MCL and LCL ... normal > ... laxity in the PCL, suspect tear" marcaba ACL
# y MCL como positivos). Cortar por estos limites separa cada hallazgo. NO se
# corta en saltos de linea crudos: los reportes envuelven una misma oracion en
# varias lineas, y cortar ahi parte "Rotura ... del menisco lateral" en dos
# (medido: hundia el recall de Lateral Meniscus).
_SEPARADOR_DURO = re.compile(r"[>•]+|(?:^|\s)\*\s")


def dividir_en_oraciones(texto: str) -> list[str]:
    """Split a report into sentences, keeping only non-trivial ones."""
    if not texto or not texto.strip():
        return []

    oraciones = []
    for bloque in _SEPARADOR_DURO.split(texto):
        if not bloque or not bloque.strip():
            continue
        for pieza in _dividir_por_puntuacion(bloque):
            limpia = " ".join(pieza.split())
            if len(limpia) >= _MINIMO_CARACTERES:
                oraciones.append(limpia)
    return oraciones


def _dividir_por_puntuacion(bloque: str) -> Iterator[str]:
    inicio = 0
    for coincidencia in _CANDIDATO.finditer(bloque):
        if not _termina_oracion(bloque, coincidencia.start()):
            continue
        yield bloque[inicio:coincidencia.end()]
        inicio = coincidencia.end()
    if inicio < len(bloque):
        yield bloque[inicio:]


def _termina_oracion(bloque: str, posicion: int) -> bool:
    """False when the period belongs to a number or a known abbreviation."""
    anterior = bloque[:posicion]
    if anterior.endswith(tuple("0123456789")):
        return False
    palabra = _ULTIMA_PALABRA.search(anterior)
    return not (palabra and palabra.group(1).lower() in ABREVIATURAS)
