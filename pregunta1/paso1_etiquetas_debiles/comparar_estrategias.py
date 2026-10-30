"""Las cuatro estrategias de traducción, medidas en una sola corrida.

El paso 1 tiene que elegir **a qué informes aplicarles NLLB** antes de extraer.
Hay cuatro opciones razonables y la diferencia entre ellas no es obvia a priori:
traducir de más puede degradar un idioma que el extractor ya leía bien.

`comparar_traduccion.py` contrasta dos (sin traducir contra traducir todo) y
`evaluar_extractor.py` mide la que corre en producción. Este script mide **las
cuatro con el mismo código y el mismo conjunto**, que es lo que hace falta para
poder ponerlas en una tabla y decir cuál gana sin mezclar corridas distintas.

**Por qué existe.** El README del paso publicaba esa tabla de cuatro filas sin un
script que la produjera, y las filas estaban corridas: le atribuía a "traducir
salvo inglés y español" el acuerdo de "traducir todo".

Usage:
    python pregunta1/paso1_etiquetas_debiles/comparar_estrategias.py
"""

from __future__ import annotations

import argparse
import csv
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
from pregunta1.paso1_etiquetas_debiles import ExtractorReglas  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.evaluacion import evaluar  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.terminos import LABELS  # noqa: E402

TRADUCIDOS = pathlib.Path(__file__).resolve().parent / "reportes_traducidos.csv"
SALIDA = pathlib.Path(__file__).resolve().parent / "comparar_estrategias.json"

#: Qué idiomas se dejan en su original en cada estrategia. El extractor lee
#: inglés y español nativos (`terminos.py`), así que son los dos candidatos a
#: quedarse sin traducir.
ESTRATEGIAS = (
    ("sin traducir", None),
    ("traducir todo", frozenset()),
    ("traducir salvo inglés", frozenset({"en"})),
    ("traducir salvo inglés y español", frozenset({"en", "es"})),
)


def cargar_traducciones(uids: set) -> dict[str, tuple[str, str]]:
    """``uid -> (idioma, reporte_en)`` para lo que haya traducido en caché."""
    if not TRADUCIDOS.is_file():
        raise FileNotFoundError(
            f"falta {TRADUCIDOS.name}. Corré primero:\n"
            f"  python pregunta1/paso1_etiquetas_debiles/traducir_reportes.py")
    salida = {}
    with TRADUCIDOS.open(encoding="utf-8") as mano:
        for fila in csv.DictReader(mano):
            if fila["StudyInstanceUID"] in uids and fila.get("reporte_en", "").strip():
                salida[fila["StudyInstanceUID"]] = (fila["idioma"],
                                                    fila["reporte_en"])
    return salida


def textos_de(oro, traducciones, nativos) -> dict[str, str]:
    """El texto que vería el extractor bajo una estrategia.

    ``nativos is None`` significa no traducir nada; si no, es el conjunto de
    idiomas que se dejan en su original.
    """
    textos = {}
    for e in oro:
        idioma, traducido = traducciones.get(e.uid, (None, None))
        usar_original = (nativos is None or traducido is None
                         or idioma in nativos)
        textos[e.uid] = e.report if usar_original else traducido
    return textos


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=pathlib.Path,
                        default=_rutas.TRAIN_CSV)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    oro = gold_studies(load_studies(args.train_csv))
    traducciones = cargar_traducciones({e.uid for e in oro})
    extractor = ExtractorReglas()
    reales = [e.positives for e in oro]

    print(f"las cuatro estrategias sobre los mismos {len(oro)} dorados\n")
    print(f"{'estrategia':<34} {'acuerdo':>8} {'F1 macro':>9} {'ruido':>7}")
    print("-" * 62)
    filas = []
    for nombre, nativos in ESTRATEGIAS:
        textos = textos_de(oro, traducciones, nativos)
        pred = [extractor.extraer(textos[e.uid]).etiquetas for e in oro]
        r = evaluar(pred, reales, LABELS)
        filas.append({"estrategia": nombre,
                      "idiomas_sin_traducir": sorted(nativos) if nativos else
                      ([] if nativos is not None else None),
                      "acuerdo_macro": round(100 * r.acuerdo_macro, 1),
                      "f1_macro": round(r.f1_macro, 3),
                      "ruido_macro": round(100 * r.ruido_macro, 1)})
        print(f"{nombre:<34} {100*r.acuerdo_macro:>7.1f}% "
              f"{r.f1_macro:>9.3f} {100*r.ruido_macro:>6.1f}%")

    mejor = max(filas, key=lambda f: f["f1_macro"])
    print("-" * 62)
    print(f"gana: {mejor['estrategia']} (F1 {mejor['f1_macro']:.3f})")
    base = next(f for f in filas if f["estrategia"] == "sin traducir")
    print(f"contra no traducir nada: {mejor['f1_macro'] - base['f1_macro']:+.3f}")

    args.salida.write_text(
        json.dumps({"n_dorados": len(oro), "estrategias": filas,
                    "mejor": mejor["estrategia"]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nescrito: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
