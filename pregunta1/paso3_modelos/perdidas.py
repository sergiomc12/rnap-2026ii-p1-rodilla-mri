"""Loss functions for the twelve-label problem.

The exam asks the ablation to compare "BCE estándar vs. pérdida focal,
dado el desbalance esperado en hallazgos raros como fractura o quiste de
Baker". Both live here so the comparison changes one argument and nothing
else.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerdidaFocal(nn.Module):
    """Focal loss for independent binary labels.

    Standard BCE lets the easy negatives dominate: in this sample Lateral
    OA is positive in 12.2% of studies, so a model that answers "no" to
    everything already looks right 88% of the time and the gradient from
    the few positives is drowned out.

    Focal loss multiplies each term by (1 - p_t)^gamma, which shrinks the
    contribution of examples the model already gets right and leaves the
    hard ones carrying the gradient.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25,
                 reduccion: str = "mean") -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduccion not in ("mean", "sum", "none"):
            raise ValueError("reduccion must be mean, sum or none")
        self.gamma = gamma
        self.alpha = alpha
        self.reduccion = reduccion

    def forward(self, logits: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, objetivo, reduction="none")
        probabilidad = torch.sigmoid(logits)
        # p_t: how confident the model is about the *correct* answer.
        p_t = probabilidad * objetivo + (1 - probabilidad) * (1 - objetivo)
        perdida = bce * (1 - p_t) ** self.gamma

        if self.alpha is not None:
            peso = self.alpha * objetivo + (1 - self.alpha) * (1 - objetivo)
            perdida = perdida * peso

        if self.reduccion == "mean":
            return perdida.mean()
        if self.reduccion == "sum":
            return perdida.sum()
        return perdida


def pesos_por_prevalencia(
    prevalencias: Sequence[float], tope: float = 10.0
) -> torch.Tensor:
    """Positive-class weights for BCE, inversely proportional to prevalence.

    A rare finding gets a larger weight so its positives are not ignored.
    The cap keeps a near-absent label from producing a weight so large that
    it destabilizes training.
    """
    pesos = []
    for prevalencia in prevalencias:
        if prevalencia <= 0:
            pesos.append(tope)
        else:
            pesos.append(min(tope, (1 - prevalencia) / prevalencia))
    return torch.tensor(pesos, dtype=torch.float32)


class PerdidaEnmascarada(nn.Module):
    """Entrena solo un subconjunto de las doce etiquetas.

    Existe para el entrenamiento por cara: medido a 5 epocas, cada plano es
    el mejor para un subconjunto distinto de hallazgos --`Synovitis` y `ACL`
    en axial, `PF OA` y `Lateral Meniscus` en coronal, las ocho restantes en
    sagital--. Un modelo que solo responde por sus etiquetas no deberia
    gastar capacidad del backbone en las otras diez.

    El promedio se toma SOLO sobre las columnas activas. El atajo obvio
    --multiplicar la perdida por una mascara de ceros y dejar
    ``reduction="mean"``-- esta mal de una forma que no se nota: dividiria
    por doce en vez de por la cantidad de activas, asi que con dos etiquetas
    activas el gradiente saldria seis veces mas chico de lo que corresponde y
    el learning rate efectivo seria otro, sin que nada lo avise.
    """

    def __init__(self, base: nn.Module, indices_activos: Sequence[int]) -> None:
        super().__init__()
        if not len(indices_activos):
            raise ValueError("hace falta al menos una etiqueta activa")
        self.base = base
        # Buffer y no tensor suelto: asi `.to(dispositivo)` lo mueve junto
        # con el modulo, en vez de quedar en CPU e indexar mal en CUDA.
        self.register_buffer(
            "indices",
            torch.tensor(sorted(set(indices_activos)), dtype=torch.long),
        )

    def forward(self, logits: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
        por_elemento = self.base(logits, objetivo)
        return por_elemento.index_select(1, self.indices).mean()


def construir_perdida(
    nombre: str,
    prevalencias: Optional[Sequence[float]] = None,
    indices_activos: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Factory used by the ablation of deliverable #4.

    Con ``indices_activos`` la perdida solo cuenta esas columnas; se usa para
    entrenar una cara con las etiquetas que mejor le salen (ver
    ``PerdidaEnmascarada``). La perdida base se construye con
    ``reduction="none"`` en ese caso, porque el promedio lo hace el envoltorio
    sobre las columnas activas y no sobre las doce.
    """
    enmascarar = indices_activos is not None
    reduccion = "none" if enmascarar else "mean"

    if nombre == "bce":
        base = nn.BCEWithLogitsLoss(reduction=reduccion)
    elif nombre == "bce_ponderada":
        if prevalencias is None:
            raise ValueError("bce_ponderada necesita las prevalencias")
        base = nn.BCEWithLogitsLoss(
            pos_weight=pesos_por_prevalencia(prevalencias), reduction=reduccion
        )
    elif nombre == "focal":
        base = PerdidaFocal(reduccion=reduccion)
    else:
        raise ValueError(f"perdida desconocida: {nombre}")

    return PerdidaEnmascarada(base, indices_activos) if enmascarar else base
