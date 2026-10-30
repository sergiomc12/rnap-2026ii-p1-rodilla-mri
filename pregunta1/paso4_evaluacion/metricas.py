"""Metrics for the twelve-label problem.

Lives in the evaluation step because that is what it is for, but training
(step 3) imports it too, to watch validation AUC between epochs.

The official competition metric is macro-averaged ROC AUC, which weights
all twelve findings equally: improving MCL, with 9 positives among the
gold studies, counts exactly as much as improving Effusion with 35. Any
summary that hides the per-label breakdown hides where the model actually
fails.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)


def auc_roc(puntajes: Sequence[float], verdad: Sequence[int]) -> Optional[float]:
    """ROC AUC for one label, or None when it is undefined.

    Computed from ranks (the Mann-Whitney U identity) rather than by
    integrating the curve, which handles ties correctly and needs no
    threshold sweep.

    Returns None when the label is all-positive or all-negative in this
    split: the AUC simply does not exist there, and reporting 0.5 would
    quietly pull the macro average toward chance.
    """
    y = np.asarray(verdad, dtype=np.int64)
    positivos, negativos = int(y.sum()), int((1 - y).sum())
    if positivos == 0 or negativos == 0:
        return None

    orden = np.argsort(np.asarray(puntajes, dtype=np.float64), kind="mergesort")
    rangos = np.empty(len(y), dtype=np.float64)
    rangos[orden] = np.arange(1, len(y) + 1, dtype=np.float64)

    # Average the ranks of tied scores, or ties would bias the result.
    valores = np.asarray(puntajes, dtype=np.float64)[orden]
    inicio = 0
    for fin in range(1, len(valores) + 1):
        if fin == len(valores) or valores[fin] != valores[inicio]:
            if fin - inicio > 1:
                rangos[orden[inicio:fin]] = rangos[orden[inicio:fin]].mean()
            inicio = fin

    suma_positivos = rangos[y == 1].sum()
    return float((suma_positivos - positivos * (positivos + 1) / 2)
                 / (positivos * negativos))


@dataclass(frozen=True)
class ResultadoAUC:
    """Per-label AUCs and their macro average."""

    por_etiqueta: dict[str, Optional[float]]

    @property
    def definidas(self) -> list[float]:
        return [v for v in self.por_etiqueta.values() if v is not None]

    @property
    def macro(self) -> float:
        valores = self.definidas
        return sum(valores) / len(valores) if valores else 0.0

    def macro_de(self, etiquetas: Sequence[str]) -> float:
        """Macro sobre un SUBCONJUNTO de etiquetas.

        Lo necesita el entrenamiento por cara: un modelo que solo entrena
        `Synovitis` y `ACL` no debe elegir su mejor epoca por el promedio de
        las doce, porque las otras diez salen de cabezas cuyo gradiente esta
        apagado --su AUC es el del azar inicial y se mueve solo por el ruido
        del backbone compartido--. Promediar eso diluiria la senal de las dos
        que importan en diez numeros que no significan nada.
        """
        valores = [self.por_etiqueta[e] for e in etiquetas
                   if self.por_etiqueta.get(e) is not None]
        return sum(valores) / len(valores) if valores else 0.0

    @property
    def n_indefinidas(self) -> int:
        return sum(1 for v in self.por_etiqueta.values() if v is None)


def auc_macro(
    puntajes: np.ndarray, verdad: np.ndarray, etiquetas: Sequence[str] = LABELS
) -> ResultadoAUC:
    """AUC per label plus the macro average. Arrays are (N, 12)."""
    puntajes, verdad = np.asarray(puntajes), np.asarray(verdad)
    if puntajes.shape != verdad.shape:
        raise ValueError("puntajes y verdad deben tener la misma forma")
    return ResultadoAUC(por_etiqueta={
        etiqueta: auc_roc(puntajes[:, i], verdad[:, i])
        for i, etiqueta in enumerate(etiquetas)
    })


@dataclass(frozen=True)
class IntervaloBootstrap:
    """A point estimate with its percentile confidence interval."""

    valor: float
    bajo: float
    alto: float
    remuestreos: int

    def __str__(self) -> str:
        return f"{self.valor:.3f} [{self.bajo:.3f}, {self.alto:.3f}]"


def bootstrap_auc_macro(
    puntajes: np.ndarray,
    verdad: np.ndarray,
    remuestreos: int = 1000,
    semilla: int = 42,
    confianza: float = 0.95,
) -> IntervaloBootstrap:
    """Confidence interval for macro AUC by resampling studies.

    This is not optional here. The gold set has 58 studies and MCL has 9
    positives, so the interval on a single AUC spans roughly +/-0.15: a
    three-point gap between two models is not a difference, and a table
    without intervals invites the reader to believe it is.
    """
    # Se convierte UNA vez, acá, en vez de en cada una de las 24 000 llamadas a
    # auc_roc que siguen (2 000 remuestreos x 12 etiquetas x 2 conversiones).
    # Además deja los arreglos como copias propias y contiguas, en lugar de
    # vistas sobre memoria que otra biblioteca administra.
    puntajes = np.ascontiguousarray(puntajes, dtype=np.float64)
    verdad = np.ascontiguousarray(verdad, dtype=np.int64)
    n = len(verdad)
    azar = random.Random(semilla)

    valores = []
    for _ in range(remuestreos):
        indices = [azar.randrange(n) for _ in range(n)]
        resultado = auc_macro(puntajes[indices], verdad[indices])
        if resultado.definidas:
            valores.append(resultado.macro)

    valores.sort()
    cola = (1 - confianza) / 2
    return IntervaloBootstrap(
        valor=auc_macro(puntajes, verdad).macro,
        bajo=valores[int(cola * len(valores))] if valores else 0.0,
        alto=valores[int((1 - cola) * len(valores))] if valores else 0.0,
        remuestreos=remuestreos,
    )


@dataclass(frozen=True)
class MetricasUmbral:
    """Precision, recall and F1 at one chosen threshold."""

    umbral: float
    precision: float
    recall: float
    f1: float


def metricas_en_umbral(
    puntajes: np.ndarray, verdad: np.ndarray, umbral: float
) -> MetricasUmbral:
    """Macro precision/recall/F1 at a threshold, as deliverable #3 asks."""
    predicho = np.asarray(puntajes) >= umbral
    verdad = np.asarray(verdad).astype(bool)

    precisiones, recalls, f1s = [], [], []
    for columna in range(verdad.shape[1]):
        p, v = predicho[:, columna], verdad[:, columna]
        tp = int((p & v).sum())
        precision = tp / max(1, int(p.sum()))
        recall = tp / max(1, int(v.sum()))
        precisiones.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)

    media = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return MetricasUmbral(umbral=umbral, precision=media(precisiones),
                          recall=media(recalls), f1=media(f1s))


def mejor_umbral(
    puntajes: np.ndarray, verdad: np.ndarray, candidatos: Optional[Sequence[float]] = None
) -> MetricasUmbral:
    """Threshold maximising macro F1 -- the justification deliverable #3 wants.

    Chosen on the validation split, never on the gold set: picking it on
    the data the result is reported over would inflate that result.
    """
    candidatos = candidatos or [i / 100 for i in range(5, 96, 5)]
    return max((metricas_en_umbral(puntajes, verdad, u) for u in candidatos),
               key=lambda m: m.f1)
