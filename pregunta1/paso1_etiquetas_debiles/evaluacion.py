"""Measuring a weak label extractor against the gold studies.

Deliverable #1 of the exam requires validating the pipeline against the
gold-labelled studies and reporting an agreement / noise rate. With 58
studies the numbers are noisy, so every rate comes with a count beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Confusion:
    """Counts for one label over one set of studies."""

    etiqueta: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def acuerdo(self) -> float:
        """Fraction of studies where extractor and radiologist agree."""
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def ruido(self) -> float:
        """The complement of agreement: the noise rate the exam asks for."""
        return 1.0 - self.acuerdo

    @property
    def precision(self) -> float:
        predichos = self.tp + self.fp
        return self.tp / predichos if predichos else 0.0

    @property
    def recall(self) -> float:
        reales = self.tp + self.fn
        return self.tp / reales if reales else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def mas(self, predicho: bool, real: bool) -> "Confusion":
        """Return a new Confusion with one more observation counted."""
        return Confusion(
            etiqueta=self.etiqueta,
            tp=self.tp + (predicho and real),
            fp=self.fp + (predicho and not real),
            fn=self.fn + (not predicho and real),
            tn=self.tn + (not predicho and not real),
        )


@dataclass(frozen=True)
class Reporte:
    """Per-label confusions plus the macro averages across labels."""

    confusiones: tuple[Confusion, ...]

    @property
    def acuerdo_macro(self) -> float:
        return self._promedio(c.acuerdo for c in self.confusiones)

    @property
    def f1_macro(self) -> float:
        return self._promedio(c.f1 for c in self.confusiones)

    @property
    def ruido_macro(self) -> float:
        return 1.0 - self.acuerdo_macro

    def por_etiqueta(self, etiqueta: str) -> Confusion:
        for confusion in self.confusiones:
            if confusion.etiqueta == etiqueta:
                return confusion
        raise KeyError(etiqueta)

    @staticmethod
    def _promedio(valores: Iterable[float]) -> float:
        lista = list(valores)
        return sum(lista) / len(lista) if lista else 0.0


def evaluar(
    predichos: Sequence[dict[str, int]],
    reales: Sequence[frozenset[str]],
    etiquetas: Sequence[str],
) -> Reporte:
    """Compare extracted labels against gold positives, study by study.

    `predichos[i]` maps every label to 0/1; `reales[i]` is the set of
    labels a radiologist marked positive for that same study.
    """
    if len(predichos) != len(reales):
        raise ValueError("predichos y reales deben tener el mismo largo")

    confusiones = {e: Confusion(etiqueta=e) for e in etiquetas}
    for prediccion, positivos in zip(predichos, reales):
        for etiqueta in etiquetas:
            confusiones[etiqueta] = confusiones[etiqueta].mas(
                predicho=bool(prediccion.get(etiqueta, 0)),
                real=etiqueta in positivos,
            )
    return Reporte(confusiones=tuple(confusiones[e] for e in etiquetas))
