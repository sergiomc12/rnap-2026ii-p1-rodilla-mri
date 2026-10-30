"""Decide whether the gap between two runs is real or sampling noise.

The per-run intervals in `evaluar.py` answer "how precise is this number".
They do not answer "is A better than B", because two overlapping intervals
can still come from a difference that holds on every resample -- the runs
are evaluated on the *same* 56 studies, so their errors are correlated.

A paired bootstrap resamples the studies once and scores both models on
that same resample, so the correlation cancels and what is left is the
interval of the difference. If it crosses zero, there is no difference to
report.

Usage:
    python pregunta1/paso4_evaluacion/comparar.py
    python pregunta1/paso4_evaluacion/comparar.py --contra resnet50_promedio
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

import torch  # noqa: E402

from pregunta1.paso3_modelos.datos import cargar_registros  # noqa: E402
from pregunta1.paso3_modelos.particion import particionar  # noqa: E402
from pregunta1.paso4_evaluacion.evaluar import (  # noqa: E402
    CACHE, CORRIDAS, SELECCION, TRAIN_CSV, etiquetas_doradas, predecir,
)
from pregunta1.paso4_evaluacion.metricas import auc_macro  # noqa: E402
from pregunta1.paso3_modelos.modelos import construir_modelo  # noqa: E402

SALIDA = _RAIZ / "pregunta1/paso4_evaluacion/comparaciones.json"


@dataclass(frozen=True)
class Comparacion:
    """The difference between two runs, with its confidence interval."""

    a: str
    b: str
    auc_a: float
    auc_b: float
    delta: float
    bajo: float
    alto: float
    probabilidad_a_mejor: float

    @property
    def significativa(self) -> bool:
        """True when the interval of the difference stays on one side of zero."""
        return self.bajo > 0 or self.alto < 0

    def __str__(self) -> str:
        marca = "SI" if self.significativa else "no"
        return (f"{self.a:30} {self.auc_a:.3f}  vs  {self.b:30} {self.auc_b:.3f}  "
                f"delta {self.delta:+.3f} [{self.bajo:+.3f}, {self.alto:+.3f}]  "
                f"P={self.probabilidad_a_mejor:.0%}  significativa={marca}")


def comparar(
    nombre_a: str, puntajes_a: np.ndarray,
    nombre_b: str, puntajes_b: np.ndarray,
    verdad: np.ndarray, remuestreos: int = 2000, semilla: int = 42,
) -> Comparacion:
    """Paired bootstrap of the macro-AUC difference between two runs."""
    azar = random.Random(semilla)
    n = len(verdad)
    deltas = []

    for _ in range(remuestreos):
        indices = [azar.randrange(n) for _ in range(n)]
        sub_verdad = verdad[indices]
        a = auc_macro(puntajes_a[indices], sub_verdad)
        b = auc_macro(puntajes_b[indices], sub_verdad)
        if a.definidas and b.definidas:
            deltas.append(a.macro - b.macro)

    deltas.sort()
    return Comparacion(
        a=nombre_a, b=nombre_b,
        auc_a=auc_macro(puntajes_a, verdad).macro,
        auc_b=auc_macro(puntajes_b, verdad).macro,
        delta=auc_macro(puntajes_a, verdad).macro - auc_macro(puntajes_b, verdad).macro,
        bajo=deltas[int(0.025 * len(deltas))],
        alto=deltas[int(0.975 * len(deltas))],
        probabilidad_a_mejor=sum(1 for d in deltas if d > 0) / len(deltas),
    )


def predicciones_de(carpeta: pathlib.Path, registros, dispositivo):
    meta = json.loads((carpeta / "historial.json").read_text(encoding="utf-8"))
    modelo = construir_modelo(meta["modelo"], pooling=meta["pooling"],
                              preentrenado=False)
    modelo.load_state_dict(torch.load(carpeta / "pesos.pt", map_location="cpu"))
    return predecir(modelo.to(dispositivo), registros, dispositivo)


# The comparisons that answer the questions the exam actually asks.
PARES = (
    ("inception_bce_atencion", "resnet50_bce_atencion",
     "entregable #2: las dos arquitecturas"),
    ("resnet50_promedio", "resnet50_bce_atencion",
     "ablacion: pooling por media vs. atencion"),
    ("resnet50_bce_atencion", "resnet50_sin_imagenet",
     "ablacion: con vs. sin pesos de ImageNet"),
    ("resnet50_focal", "resnet50_bce_atencion",
     "ablacion: focal vs. BCE (con atencion)"),
    ("resnet50_promedio_focal", "resnet50_promedio",
     "ablacion: focal vs. BCE (con media)"),
    ("resnet50_promedio_ponderada", "resnet50_promedio",
     "desbalance: BCE ponderada vs. BCE"),
)

# El plano va aparte porque no se puede meter en PARES: las dos corridas leen
# cachés distintos (sagital y coronal), de modo que no comparten los tensores
# de entrada aunque sí los estudios. Durante un tiempo esta comparación figuró
# en el informe sin que ningún script la calculara; está acá para que el número
# salga del pipeline y no de una corrida suelta que nadie puede repetir.
SELECCION_CORONAL = _RAIZ / "pregunta1/paso2_preprocesamiento/estudios_coronal.csv"
CACHE_CORONAL = _RAIZ / "pregunta1/paso2_preprocesamiento/cache_coronal"
PAR_PLANOS = ("resnet50_promedio", "resnet50_promedio_coronal",
              "ablacion: plano sagital vs. coronal")


def predicciones_entre_planos(
    dorados_sagital, dispositivo,
    seleccion_coronal: pathlib.Path = SELECCION_CORONAL,
    cache_coronal: pathlib.Path = CACHE_CORONAL,
) -> Optional[dict]:
    """Fase de GPU de la comparación entre planos.

    Solo entran los estudios dorados que existen en los **dos** planos: un
    estudio sin serie coronal no se puede comparar consigo mismo, y mezclar
    subconjuntos distintos convertiría una diferencia de plano en una
    diferencia de muestra.

    Va separada del bootstrap por el mismo motivo que en ``evaluar.py``:
    intercalar inferencia en GPU con los bucles de NumPy del remuestreo
    aborta el proceso con una violación de acceso.
    """
    nombre_a, nombre_b, _ = PAR_PLANOS
    carpeta_a, carpeta_b = CORRIDAS / nombre_a, CORRIDAS / nombre_b
    if not (carpeta_b / "historial.json").is_file():
        print(f"falta la corrida {nombre_b}: se omite la comparación de planos")
        return None
    if not seleccion_coronal.is_file() or not cache_coronal.is_dir():
        print("falta el conjunto coronal: se omite la comparación de planos")
        return None

    registros_coronal = cargar_registros(seleccion_coronal, cache_coronal,
                                         etiquetas_doradas(TRAIN_CSV))
    por_uid = {r.uid: r for r in particionar(registros_coronal).prueba}
    comunes = [r for r in dorados_sagital if r.uid in por_uid]
    if len(comunes) < 10:
        print(f"solo {len(comunes)} estudios en ambos planos: no alcanza")
        return None

    puntajes_a, verdad = predicciones_de(carpeta_a, comunes, dispositivo)
    puntajes_b, verdad_b = predicciones_de(
        carpeta_b, [por_uid[r.uid] for r in comunes], dispositivo)
    # Mismo orden de estudios en los dos: si las verdades no coinciden, el
    # emparejamiento está roto y el intervalo que saldría no significaría nada.
    if not np.array_equal(verdad, verdad_b):
        raise RuntimeError("las etiquetas no coinciden entre planos: "
                           "el pareado quedaría desalineado")

    return {"puntajes_a": puntajes_a, "puntajes_b": puntajes_b,
            "verdad": verdad, "estudios": len(comunes),
            "de_total": len(dorados_sagital)}


def comparar_planos(datos: dict, remuestreos: int) -> dict:
    """Fase de métricas de la comparación entre planos."""
    nombre_a, nombre_b, descripcion = PAR_PLANOS
    print(f"--- {descripcion} ---")
    print(f"    {datos['estudios']} de {datos['de_total']} estudios dorados "
          f"tienen ambos planos")
    comparacion = comparar(nombre_a, datos["puntajes_a"],
                           nombre_b, datos["puntajes_b"], datos["verdad"],
                           remuestreos=remuestreos)
    print(f"    {comparacion}\n")
    return {"descripcion": descripcion, **comparacion.__dict__,
            "significativa": comparacion.significativa,
            "estudios": datos["estudios"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remuestreos", type=int, default=2000)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registros = cargar_registros(SELECCION, CACHE, etiquetas_doradas(TRAIN_CSV))
    dorados = particionar(registros).prueba

    nombres = sorted({n for par in PARES for n in par[:2]})
    predicciones, verdad = {}, None
    for nombre in nombres:
        carpeta = CORRIDAS / nombre
        if not (carpeta / "historial.json").is_file():
            print(f"falta la corrida {nombre}")
            return 1
        predicciones[nombre], verdad = predicciones_de(carpeta, dorados, dispositivo)

    # Toda la inferencia va junta, incluida la de la comparacion entre planos,
    # y recien despues empiezan los remuestreos. Ver predicciones_entre_planos.
    datos_planos = predicciones_entre_planos(dorados, dispositivo)
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()

    print(f"BOOTSTRAP PAREADO · {len(dorados)} estudios dorados · "
          f"{args.remuestreos} remuestreos\n")

    resultados = []
    for a, b, descripcion in PARES:
        comparacion = comparar(a, predicciones[a], b, predicciones[b], verdad,
                               remuestreos=args.remuestreos)
        print(f"--- {descripcion} ---")
        print(f"    {comparacion}\n")
        resultados.append({"descripcion": descripcion, **comparacion.__dict__,
                           "significativa": comparacion.significativa})

    if datos_planos is not None:
        resultados.append(comparar_planos(datos_planos, args.remuestreos))

    args.salida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    significativas = sum(1 for r in resultados if r["significativa"])
    print(f"{significativas} de {len(resultados)} diferencias son significativas")
    print(f"guardado en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
