"""¿Promediar varios modelos mejora el AUC sobre los dorados?

Es la pregunta más barata de todo el proyecto: los modelos ya están entrenados,
así que ensamblar no cuesta una sola época. Pero también es la más fácil de
responder mal, porque un ensamble casi siempre sube un poco y con 56 estudios
de prueba "un poco" no se distingue de cero.

Se evalúan dos formas de combinar, que no son equivalentes:

* **media de probabilidades** — la directa. Un modelo mal calibrado que predice
  todo cerca de 0.9 arrastra la media.
* **media de rangos** — cada modelo se normaliza a su propio orden antes de
  promediar. El AUC solo mira el orden, así que esto neutraliza las diferencias
  de calibración entre modelos; es lo que se usa habitualmente en competencia.

Y se reporta la **correlación entre los modelos**: un ensamble solo compra algo
cuando sus miembros se equivocan en casos distintos. Tres modelos que aciertan
y fallan juntos promedian a lo mismo que cualquiera de ellos.

Usage:
    python pregunta1/paso4_evaluacion/ensamblar.py --corrida A --corrida B --corrida C
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from typing import Optional, Sequence

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

import torch  # noqa: E402

from pregunta1.paso3_modelos.datos import cargar_registros  # noqa: E402
from pregunta1.paso3_modelos.particion import particionar  # noqa: E402
from pregunta1.paso4_evaluacion.comparar import predicciones_de  # noqa: E402
from pregunta1.paso4_evaluacion.evaluar import (  # noqa: E402
    CACHE, CORRIDAS, SELECCION, TRAIN_CSV, etiquetas_doradas,
)
from pregunta1.paso4_evaluacion.metricas import (  # noqa: E402
    auc_macro, bootstrap_auc_macro,
)

SALIDA = _RAIZ / "pregunta1/paso4_evaluacion/ensamble.json"


def a_rangos(puntajes: np.ndarray) -> np.ndarray:
    """Convierte cada columna a rangos en [0, 1].

    El AUC solo depende del orden dentro de cada etiqueta, así que esto no
    cambia el AUC de un modelo suelto —pero sí cambia cómo se combinan dos.
    """
    salida = np.empty_like(puntajes, dtype=np.float64)
    n = puntajes.shape[0]
    for columna in range(puntajes.shape[1]):
        orden = np.argsort(puntajes[:, columna], kind="mergesort")
        rangos = np.empty(n, dtype=np.float64)
        rangos[orden] = np.arange(1, n + 1, dtype=np.float64)
        salida[:, columna] = rangos / n
    return salida


def correlacion_media(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación media entre dos modelos, etiqueta por etiqueta."""
    valores = []
    for columna in range(a.shape[1]):
        x, y = a[:, columna], b[:, columna]
        if x.std() == 0 or y.std() == 0:
            continue
        valores.append(float(np.corrcoef(x, y)[0, 1]))
    return sum(valores) / len(valores) if valores else float("nan")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrida", action="append", dest="corridas",
                        required=True, help="repetible: los modelos a combinar")
    parser.add_argument("--seleccion", type=pathlib.Path, default=SELECCION)
    parser.add_argument("--cache", type=pathlib.Path, default=CACHE)
    parser.add_argument("--remuestreos", type=int, default=2000)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--dispositivo", default="cpu",
                        help="cpu por defecto: son 56 estudios, y evita chocar "
                             "con un entrenamiento que esté usando la GPU")
    args = parser.parse_args(argv)

    dispositivo = torch.device(args.dispositivo)
    dorados = particionar(cargar_registros(
        args.seleccion, args.cache, etiquetas_doradas(TRAIN_CSV))).prueba
    print(f"ENSAMBLE SOBRE {len(dorados)} ESTUDIOS DORADOS\n")

    puntajes, verdad = {}, None
    for nombre in args.corridas:
        carpeta = CORRIDAS / nombre
        if not (carpeta / "historial.json").is_file():
            print(f"falta la corrida {nombre}")
            return 1
        puntajes[nombre], verdad = predicciones_de(carpeta, dorados, dispositivo)

    informe: dict = {"miembros": {}, "correlaciones": {}}
    print(f"{'modelo':44} {'AUC':>8}")
    for nombre, p in puntajes.items():
        valor = auc_macro(p, verdad).macro
        informe["miembros"][nombre] = valor
        print(f"{nombre:44} {valor:>8.4f}")

    print(f"\ncorrelación entre miembros (más baja = más complementarios):")
    for a, b in itertools.combinations(args.corridas, 2):
        r = correlacion_media(puntajes[a], puntajes[b])
        informe["correlaciones"][f"{a}|{b}"] = r
        print(f"  {a[:26]:26} vs {b[:26]:26} {r:>6.3f}")

    matriz = np.stack([puntajes[n] for n in args.corridas])
    combinaciones = {
        "media_probabilidades": matriz.mean(axis=0),
        "media_rangos": np.stack([a_rangos(m) for m in matriz]).mean(axis=0),
    }

    mejor_miembro = max(informe["miembros"].values())
    print(f"\n{'combinación':24} {'AUC':>8} {'IC95%':>20} {'vs mejor miembro':>18}")
    for nombre, combinado in combinaciones.items():
        intervalo = bootstrap_auc_macro(combinado, verdad,
                                        remuestreos=args.remuestreos)
        informe[nombre] = {"auc": intervalo.valor,
                           "ic95": [intervalo.bajo, intervalo.alto]}
        print(f"{nombre:24} {intervalo.valor:>8.4f} "
              f"[{intervalo.bajo:.4f}, {intervalo.alto:.4f}] "
              f"{intervalo.valor - mejor_miembro:>+18.4f}")

    informe["mejor_miembro"] = mejor_miembro
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nescrito en {args.salida}")
    print("un ensamble casi siempre sube un poco; con 56 estudios de prueba "
          "'un poco' no se distingue de cero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
