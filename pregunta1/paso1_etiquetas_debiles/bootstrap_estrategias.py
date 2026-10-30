"""¿La mejora por traducir sobrevive a tener solo 58 estudios dorados?

Con n=58, una diferencia de F1 macro de 7 puntos puede ser ruido de muestreo. Lo
que decide es si el intervalo de la **diferencia** cruza cero, y para eso el
remuestreo tiene que ser **pareado**: los mismos estudios para las dos
estrategias, de modo que el ruido del conjunto se cancele en vez de sumarse dos
veces. Comparar dos intervalos sueltos es el error clásico acá, y con 58 casos
los dos intervalos se solapan casi seguro aunque la diferencia sea real.

**Se remuestrea sobre estudios, nunca sobre etiquetas.** Las doce etiquetas de un
mismo informe están correlacionadas (un informe que menciona derrame suele
mencionar sinovitis), así que tratarlas como 696 observaciones independientes
daría un intervalo artificialmente angosto.

**Por qué existe este archivo.** El README del paso publicaba un intervalo
bootstrap y un `P(hibrido mejor)` sin ningún script que los produjera.

Usage:
    python pregunta1/paso1_etiquetas_debiles/bootstrap_estrategias.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
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
from pregunta1.paso1_etiquetas_debiles.comparar_estrategias import (  # noqa: E402
    ESTRATEGIAS, cargar_traducciones, textos_de,
)
from pregunta1.paso1_etiquetas_debiles.evaluacion import evaluar  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.terminos import LABELS  # noqa: E402

SALIDA = pathlib.Path(__file__).resolve().parent / "bootstrap_estrategias.json"


def f1_macro(pred: Sequence[dict], reales: Sequence) -> float:
    return evaluar(list(pred), list(reales), LABELS).f1_macro


def percentil(valores: Sequence[float], q: float) -> float:
    orden = sorted(valores)
    if not orden:
        return float("nan")
    pos = q * (len(orden) - 1)
    bajo = int(pos)
    alto = min(bajo + 1, len(orden) - 1)
    return orden[bajo] + (orden[alto] - orden[bajo]) * (pos - bajo)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=pathlib.Path,
                        default=_rutas.TRAIN_CSV)
    parser.add_argument("--remuestreos", type=int, default=2000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    oro = gold_studies(load_studies(args.train_csv))
    traducciones = cargar_traducciones({e.uid for e in oro})
    extractor = ExtractorReglas()
    reales = [e.positives for e in oro]

    # Se predice UNA vez por estrategia y después se remuestrean los índices:
    # volver a correr el extractor dentro del bucle sería 2 000 veces más caro
    # y daría exactamente lo mismo.
    pred = {}
    for nombre, nativos in ESTRATEGIAS:
        textos = textos_de(oro, traducciones, nativos)
        pred[nombre] = [extractor.extraer(textos[e.uid]).etiquetas for e in oro]

    base, mejor = "sin traducir", "traducir salvo inglés y español"
    obs_base, obs_mejor = f1_macro(pred[base], reales), f1_macro(pred[mejor], reales)
    print(f"{len(oro)} dorados · {args.remuestreos} remuestreos pareados "
          f"sobre ESTUDIOS\n")
    print(f"  {base:<32} F1 {obs_base:.3f}")
    print(f"  {mejor:<32} F1 {obs_mejor:.3f}")
    print(f"  delta observado                   {obs_mejor - obs_base:+.3f}\n")

    rng = random.Random(args.semilla)
    n = len(oro)
    db, dm, dd = [], [], []
    for _ in range(args.remuestreos):
        idx = [rng.randrange(n) for _ in range(n)]
        r = [reales[i] for i in idx]
        a = f1_macro([pred[base][i] for i in idx], r)
        b = f1_macro([pred[mejor][i] for i in idx], r)
        db.append(a)
        dm.append(b)
        dd.append(b - a)

    gana = sum(1 for d in dd if d > 0) / len(dd)
    r = {
        "n_dorados": n, "remuestreos": args.remuestreos, "semilla": args.semilla,
        "base": {"estrategia": base, "f1": round(obs_base, 3),
                 "ic95": [round(percentil(db, .025), 3),
                          round(percentil(db, .975), 3)]},
        "mejor": {"estrategia": mejor, "f1": round(obs_mejor, 3),
                  "ic95": [round(percentil(dm, .025), 3),
                           round(percentil(dm, .975), 3)]},
        "delta": {"valor": round(obs_mejor - obs_base, 3),
                  "ic95": [round(percentil(dd, .025), 3),
                           round(percentil(dd, .975), 3)],
                  "p_mejor_gana": round(100 * gana, 1)},
    }
    r["delta"]["significativa"] = bool(r["delta"]["ic95"][0] > 0)

    print(f"F1 {base:<24} {r['base']['f1']:.3f}  IC95% "
          f"[{r['base']['ic95'][0]:.3f}, {r['base']['ic95'][1]:.3f}]")
    print(f"F1 {mejor:<24} {r['mejor']['f1']:.3f}  IC95% "
          f"[{r['mejor']['ic95'][0]:.3f}, {r['mejor']['ic95'][1]:.3f}]")
    print(f"delta pareado            {r['delta']['valor']:+.3f}  IC95% "
          f"[{r['delta']['ic95'][0]:+.3f}, {r['delta']['ic95'][1]:+.3f}]")
    print(f"P(traducir gana)         {r['delta']['p_mejor_gana']:.1f}%")
    print(f"\n{'SIGNIFICATIVA' if r['delta']['significativa'] else 'NO significativa'}"
          f": el intervalo de la diferencia "
          f"{'no cruza' if r['delta']['significativa'] else 'CRUZA'} cero.")
    print("Los dos intervalos sueltos se solapan; por eso se mira el pareado.")

    args.salida.write_text(json.dumps(r, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nescrito: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
