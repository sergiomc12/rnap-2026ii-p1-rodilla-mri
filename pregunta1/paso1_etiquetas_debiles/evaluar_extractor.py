"""Measure the rule-based extractor against the 58 gold studies.

Usage:
    python pregunta1/paso1_etiquetas_debiles/evaluar_extractor.py
"""

from __future__ import annotations

import csv
import pathlib
import sys
import argparse
from collections import defaultdict

# pregunta1/ entra al sys.path para poder importar esta pregunta como
# paquete: los scripts se corren por ruta, no con -m. Las rutas de datos
# salen de pregunta1/rutas.py: siempre dentro de pregunta1/datos/.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

from pregunta1 import rutas as _rutas  # noqa: E402

from pregunta1.paso1_etiquetas_debiles import LABELS, ExtractorReglas  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.evaluacion import evaluar  # noqa: E402
from pregunta1.estudios import gold_studies, load_studies  # noqa: E402

TRAIN_CSV = _rutas.TRAIN_CSV


def detectar_idioma(texto: str) -> str:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(texto[:600])
    except Exception:
        return "?"


def particionar_dorados(oro, semilla: int = 42):
    """Parte los dorados en desarrollo y reservados.

    Mejorar el extractor midiendo contra los 58 estudios y volviendo a
    ajustarlo contra los mismos 58 lo ajusta a ellos: el acuerdo que se
    reporta deja de medir lo que dice medir. La mitad de desarrollo se puede
    mirar cuantas veces haga falta; la reservada se toca UNA vez, al final.
    """
    import random
    mezclados = sorted(oro, key=lambda s: s.uid)
    random.Random(semilla).shuffle(mezclados)
    corte = len(mezclados) // 2
    return mezclados[:corte], mezclados[corte:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parte", choices=["desarrollo", "reservados", "todos"],
                        default="todos",
                        help="sobre que mitad de los dorados medir")
    parser.add_argument("--sin-traducir", action="store_true",
                        help="medir sobre el informe original en vez del texto "
                             "que la produccion realmente lee")
    args = parser.parse_args()

    todos = gold_studies(load_studies(TRAIN_CSV))
    desarrollo, reservados = particionar_dorados(todos)
    oro = {"desarrollo": desarrollo, "reservados": reservados,
           "todos": todos}[args.parte]
    print(f"[parte: {args.parte}]")
    extractor = ExtractorReglas()

    # La produccion lee el informe TRADUCIDO para todo idioma que no sea
    # ingles ni espanol (estrategia hibrida del paso 1). Medir sobre el
    # original mediria un extractor que no es el que corre, y cualquier
    # arreglo del lexico sobre terminologia traducida resultaria invisible.
    textos = {e.uid: e.report for e in oro}
    if not args.sin_traducir:
        textos.update(_textos_de_produccion({e.uid for e in oro}))

    predichos = [extractor.extraer(textos[s.uid]).etiquetas for s in oro]
    reales = [s.positives for s in oro]
    reporte = evaluar(predichos, reales, LABELS)

    print(f"EXTRACTOR POR REGLAS vs {len(oro)} ESTUDIOS DORADOS\n")
    print(f"{'etiqueta':18} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} "
          f"{'acuerdo':>8} {'prec':>6} {'rec':>6} {'F1':>6}")
    print("-" * 72)
    for etiqueta in LABELS:
        c = reporte.por_etiqueta(etiqueta)
        print(f"{etiqueta:18} {c.tp:>3} {c.fp:>3} {c.fn:>3} {c.tn:>3} "
              f"{c.acuerdo:>7.1%} {c.precision:>6.2f} {c.recall:>6.2f} {c.f1:>6.2f}")
    print("-" * 72)
    print(f"{'MACRO':18} {'':>15} {reporte.acuerdo_macro:>7.1%} "
          f"{'':>6} {'':>6} {reporte.f1_macro:>6.2f}")
    print(f"\nTASA DE RUIDO ESTIMADA (macro): {reporte.ruido_macro:.1%}")

    _por_idioma(oro, predichos, reales)
    return 0


def _textos_de_produccion(uids: set) -> dict:
    """El texto traducido, para los idiomas que la produccion traduce.

    La ruta se resuelve **contra este archivo**, no contra el directorio de
    trabajo. Con la ruta relativa que habia antes, correr el script desde
    cualquier lugar que no fuera la raiz del proyecto no encontraba el CSV y
    devolvia ``{}`` **en silencio**: la medicion pasaba a ser sobre los
    informes originales y reportaba numeros peores sin decir por que. Eso
    importa mas ahora que ``pregunta1/`` se comparte sola.
    """
    ruta = pathlib.Path(__file__).resolve().parent / "reportes_traducidos.csv"
    if not ruta.is_file():
        print(f"AVISO: no esta {ruta.name}; se mide sobre los informes "
              f"ORIGINALES.\n  Corre primero traducir_reportes.py, o pasa "
              f"--sin-traducir para que esto sea deliberado.")
        return {}
    salida = {}
    with ruta.open(encoding="utf-8") as mano:
        for fila in csv.DictReader(mano):
            if fila["StudyInstanceUID"] not in uids:
                continue
            if fila["idioma"] in ("en", "es"):
                continue
            if fila.get("reporte_en", "").strip():
                salida[fila["StudyInstanceUID"]] = fila["reporte_en"]
    return salida


def _por_idioma(oro, predichos, reales) -> None:
    """The per-language table that feeds deliverable #7."""
    grupos = defaultdict(list)
    for indice, estudio in enumerate(oro):
        grupos[detectar_idioma(estudio.report)].append(indice)

    print(f"\n{'idioma':8} {'n':>4} {'acuerdo':>9} {'F1 macro':>9}")
    print("-" * 34)
    for idioma, indices in sorted(grupos.items(), key=lambda kv: -len(kv[1])):
        sub = evaluar([predichos[i] for i in indices],
                      [reales[i] for i in indices], LABELS)
        print(f"{idioma:8} {len(indices):>4} {sub.acuerdo_macro:>8.1%} "
              f"{sub.f1_macro:>9.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
