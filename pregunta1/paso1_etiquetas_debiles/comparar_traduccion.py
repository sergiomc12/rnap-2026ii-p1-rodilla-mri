"""Measure what translating the reports buys, against the 58 gold studies.

Deliverable #1 asks for an agreement/noise rate; this script reports it
before and after translation, overall and per language. That same table is
the evidence for the ablation of deliverable #4 and for the linguistic bias
discussion of deliverable #7.

The 30 non-English gold studies are the only direct evidence available:
the 1,989 non-English studies without gold labels can never be checked.

Usage:
    python pregunta1/paso1_etiquetas_debiles/comparar_traduccion.py
"""

from __future__ import annotations

import csv
import pathlib
import sys
from collections import defaultdict
from typing import Optional, Sequence

# pregunta1/ entra al sys.path para poder importar esta pregunta como
# paquete: los scripts se corren por ruta, no con -m. Las rutas de datos
# salen de pregunta1/rutas.py: siempre dentro de pregunta1/datos/.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

from pregunta1 import rutas as _rutas  # noqa: E402

from pregunta1.paso1_etiquetas_debiles import LABELS, ExtractorReglas  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.evaluacion import Reporte, evaluar  # noqa: E402
from pregunta1.estudios import gold_studies, load_studies  # noqa: E402

TRAIN_CSV = _rutas.TRAIN_CSV
TRADUCIDOS = _RAIZ / "pregunta1/paso1_etiquetas_debiles/reportes_traducidos.csv"


def cargar_traducciones(ruta: pathlib.Path) -> dict[str, dict[str, str]]:
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Falta {ruta}. Corré primero traducir_reportes.py"
        )
    with ruta.open(encoding="utf-8", newline="") as handle:
        return {fila["StudyInstanceUID"]: fila for fila in csv.DictReader(handle)}


def medir(reportes: Sequence[str], reales: Sequence[frozenset[str]]) -> Reporte:
    extractor = ExtractorReglas()
    return evaluar([extractor.extraer(r).etiquetas for r in reportes], reales, LABELS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    oro = gold_studies(load_studies(TRAIN_CSV))
    traducciones = cargar_traducciones(TRADUCIDOS)
    reales = [s.positives for s in oro]

    originales = [s.report for s in oro]
    traducidos = [traducciones[s.uid]["reporte_en"] for s in oro]
    idiomas = [traducciones[s.uid]["idioma"] for s in oro]

    antes, despues = medir(originales, reales), medir(traducidos, reales)

    print(f"IMPACTO DE LA TRADUCCION · {len(oro)} ESTUDIOS DORADOS\n")
    print(f"{'':18} {'ACUERDO':>18}   {'F1 MACRO':>18}")
    print(f"{'etiqueta':18} {'antes':>8} {'despues':>9}   {'antes':>8} {'despues':>9}")
    print("-" * 60)
    for etiqueta in LABELS:
        a, d = antes.por_etiqueta(etiqueta), despues.por_etiqueta(etiqueta)
        print(f"{etiqueta:18} {a.acuerdo:>7.1%} {d.acuerdo:>9.1%}   "
              f"{a.f1:>8.2f} {d.f1:>9.2f}")
    print("-" * 60)
    print(f"{'MACRO':18} {antes.acuerdo_macro:>7.1%} {despues.acuerdo_macro:>9.1%}   "
          f"{antes.f1_macro:>8.2f} {despues.f1_macro:>9.2f}")
    print(f"\nRUIDO: {antes.ruido_macro:.1%} -> {despues.ruido_macro:.1%}")

    _por_idioma(idiomas, originales, traducidos, reales)
    return 0


def _por_idioma(idiomas, originales, traducidos, reales) -> None:
    grupos = defaultdict(list)
    for indice, idioma in enumerate(idiomas):
        grupos[idioma].append(indice)

    print(f"\n{'idioma':8} {'n':>4} {'F1 antes':>10} {'F1 despues':>12} {'delta':>8}")
    print("-" * 46)
    for idioma, indices in sorted(grupos.items(), key=lambda kv: -len(kv[1])):
        sub = lambda fuente: medir([fuente[i] for i in indices],
                                   [reales[i] for i in indices])
        a, d = sub(originales).f1_macro, sub(traducidos).f1_macro
        print(f"{idioma:8} {len(indices):>4} {a:>10.2f} {d:>12.2f} {d-a:>+8.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
