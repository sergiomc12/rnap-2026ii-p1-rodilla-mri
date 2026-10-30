"""El baseline que justifica construir el extractor: buscar el término, nada más.

Sirve para una sola cosa, y es la que pide el entregable #1: **mostrar con
números por qué un buscador de palabras clave no alcanza** en informes
radiológicos. La diferencia con `extractor.py` es exactamente una: acá no hay
análisis de negación. Todo lo demás —los mismos términos de `terminos.py`, el
mismo partido en oraciones, los mismos 58 dorados **y el mismo texto traducido
que lee la producción**— se mantiene igual, así que la comparación aísla el
efecto de manejar la negación y no mide otra cosa.

**El texto traducido no es un detalle.** Una primera versión de este script leía
el informe original mientras `evaluar_extractor.py` leía el traducido, y el
baseline salía con **mejor** recall que el extractor en ACL (0.54 contra 0.67),
que es imposible si lo único que se le quita es la negación: sacar negación solo
puede agregar positivos, nunca quitarlos. Ese imposible era la señal de que la
comparación mezclaba dos cambios.

**Por qué este archivo existe.** El README de este paso publicaba una tabla de
falsos positivos del baseline sin ningún script que la produjera: los números no
se podían auditar ni reproducir. Ahora sí.

Usage:
    python pregunta1/paso1_etiquetas_debiles/baseline_palabras.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

# pregunta1/ entra al sys.path para poder importar esta pregunta como
# paquete: los scripts se corren por ruta, no con -m. Las rutas de datos
# salen de pregunta1/rutas.py: siempre dentro de pregunta1/datos/.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

from pregunta1 import rutas as _rutas  # noqa: E402
from pregunta1.estudios import gold_studies, load_studies  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.evaluar_extractor import (  # noqa: E402
    _textos_de_produccion,
)
from pregunta1.paso1_etiquetas_debiles.oraciones import (  # noqa: E402
    dividir_en_oraciones,
)
from pregunta1.paso1_etiquetas_debiles.terminos import (  # noqa: E402
    LABELS, TERMINOS,
)

SALIDA = pathlib.Path(__file__).resolve().parent / "baseline_palabras.json"


def extraer_sin_negacion(reporte: str) -> dict[str, int]:
    """Marca 1 en cuanto el término aparece, sin mirar si está negado.

    Se conserva la regla de que una etiqueta estructural necesita estructura
    **y** patología en la misma oración: quitar eso además de la negación
    mezclaría dos cambios en una sola medición.
    """
    etiquetas = {e: 0 for e in LABELS}
    for oracion in dividir_en_oraciones(reporte):
        for etiqueta, t in TERMINOS.items():
            if not t.estructura.search(oracion):
                continue
            if t.patologia is not None and not t.patologia.search(oracion):
                continue
            etiquetas[etiqueta] = 1
    return etiquetas


def medir(oro, textos: dict) -> dict:
    cache = {e.uid: extraer_sin_negacion(textos[e.uid]) for e in oro}
    filas = []
    for etiqueta in LABELS:
        tp = fp = fn = tn = 0
        for estudio in oro:
            real = estudio.labels[etiqueta]
            pred = cache[estudio.uid][etiqueta]
            if real and pred:
                tp += 1
            elif not real and pred:
                fp += 1
            elif real and not pred:
                fn += 1
            else:
                tn += 1
        n = max(1, tp + fp + fn + tn)
        filas.append({
            "etiqueta": etiqueta, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "acuerdo": round(100 * (tp + tn) / n, 1),
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        })
    return {"n_dorados": len(oro), "etiquetas": filas}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=pathlib.Path,
                        default=_rutas.TRAIN_CSV)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--sin-traducir", action="store_true",
                        help="medir sobre el informe original; deja de ser "
                             "comparable con evaluar_extractor.py")
    args = parser.parse_args(argv)

    oro = gold_studies(load_studies(args.train_csv))
    # El mismo texto que lee la produccion, para que la unica diferencia con
    # el extractor sea la negacion (ver el docstring).
    textos = {e.uid: e.report for e in oro}
    if not args.sin_traducir:
        textos.update(_textos_de_produccion({e.uid for e in oro}))
    r = medir(oro, textos)
    r["sobre_texto_traducido"] = not args.sin_traducir

    print(f"baseline de palabras clave, SIN negacion, sobre {r['n_dorados']} "
          f"dorados")
    print(f"texto: {'traducido (igual que la produccion)' if not args.sin_traducir else 'ORIGINAL (no comparable)'}\n")
    print(f"{'etiqueta':<18} {'FP':>4} {'FN':>4} {'acuerdo':>8} "
          f"{'prec':>6} {'rec':>6}")
    print("-" * 52)
    for f in sorted(r["etiquetas"], key=lambda x: -x["fp"]):
        pr = f"{f['precision']:.2f}" if f["precision"] is not None else "-"
        re_ = f"{f['recall']:.2f}" if f["recall"] is not None else "-"
        print(f"{f['etiqueta']:<18} {f['fp']:>4} {f['fn']:>4} "
              f"{f['acuerdo']:>7.1f}% {pr:>6} {re_:>6}")
    macro = sum(f["acuerdo"] for f in r["etiquetas"]) / len(r["etiquetas"])
    print("-" * 52)
    print(f"{'MACRO':<18} {sum(f['fp'] for f in r['etiquetas']):>4} "
          f"{sum(f['fn'] for f in r['etiquetas']):>4} {macro:>7.1f}%")
    r["acuerdo_macro"] = round(macro, 1)
    args.salida.write_text(json.dumps(r, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nescrito: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
