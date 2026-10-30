"""Splitting the studies, under the rule that governs the whole project.

    Decisions are made with cross-validation over the WEAK labels.
    The 58 gold studies are touched ONCE, at the end, for the reported number.

The gold set is 58 studies, with 9 positives for MCL. Selecting a model
against it would not measure generalization, it would measure how well the
selection overfit 58 examples -- and the result reported afterwards would
be on that same data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .datos import Registro


@dataclass(frozen=True)
class Particion:
    """Train / validation / test, with test reserved for the gold studies."""

    entrenamiento: tuple[Registro, ...]
    validacion: tuple[Registro, ...]
    prueba: tuple[Registro, ...]

    def resumen(self) -> str:
        return (f"entrenamiento={len(self.entrenamiento)} "
                f"validacion={len(self.validacion)} "
                f"prueba(dorados)={len(self.prueba)}")


def particionar(
    registros: Sequence[Registro], fraccion_validacion: float = 0.2, semilla: int = 42
) -> Particion:
    """Hold out every gold study as test, split the rest by study.

    Splitting by study and not by slice group matters: two groups of the
    same knee landing on both sides of the split would leak.
    """
    if not 0.0 < fraccion_validacion < 1.0:
        raise ValueError("fraccion_validacion must be in (0, 1)")

    dorados = [r for r in registros if r.es_dorado]
    resto = [r for r in registros if not r.es_dorado]

    azar = random.Random(semilla)
    mezclados = list(resto)
    azar.shuffle(mezclados)

    corte = int(len(mezclados) * fraccion_validacion)
    return Particion(
        entrenamiento=tuple(mezclados[corte:]),
        validacion=tuple(mezclados[:corte]),
        prueba=tuple(dorados),
    )


def pliegues(
    registros: Sequence[Registro], n_pliegues: int = 5, semilla: int = 42
) -> list[Particion]:
    """K-fold over the weak-labelled studies, gold always held out as test."""
    if n_pliegues < 2:
        raise ValueError("n_pliegues must be >= 2")

    dorados = tuple(r for r in registros if r.es_dorado)
    resto = [r for r in registros if not r.es_dorado]

    azar = random.Random(semilla)
    mezclados = list(resto)
    azar.shuffle(mezclados)

    bloques = [mezclados[i::n_pliegues] for i in range(n_pliegues)]
    particiones = []
    for indice in range(n_pliegues):
        entrenamiento = [r for j, bloque in enumerate(bloques)
                         if j != indice for r in bloque]
        particiones.append(Particion(
            entrenamiento=tuple(entrenamiento),
            validacion=tuple(bloques[indice]),
            prueba=dorados,
        ))
    return particiones
