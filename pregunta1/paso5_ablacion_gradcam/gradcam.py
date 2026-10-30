"""Grad-CAM for the multi-label knee classifier.

Deliverable #5 asks for activation maps on at least three findings and a
discussion of whether the highlighted regions are anatomically plausible.
None of the course notebooks implement Grad-CAM -- verified, zero
`register_forward_hook` in all four -- so this is written from scratch
following Selvaraju et al.

The method: take the gradient of one label's logit with respect to the last
convolutional feature map, average it over each channel to get that
channel's importance, combine the channels with those weights, and keep the
positive part. What survives is where the evidence for that label was.

Two wrinkles this architecture adds:

* The prediction is per study, not per slice group, so the map is computed
  for a chosen group. The attention weights say which group the model
  leaned on, and that is the group worth looking at -- the two analyses
  compose: attention says *which slice*, Grad-CAM says *where inside it*.
* There are twelve independent sigmoid outputs, so there is one map per
  label, not one per study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MapaActivacion:
    """One Grad-CAM map and the context needed to read it."""

    mapa: np.ndarray          # (H, W) in [0, 1]
    etiqueta: str
    probabilidad: float
    grupo: int
    atencion: np.ndarray      # (G,) attention weights over slice groups

    @property
    def grupo_mas_atendido(self) -> int:
        return int(np.argmax(self.atencion))


class GradCAM:
    """Activation maps from the last convolutional stage of the encoder.

    The hooks stay registered for the life of the object and are removed by
    `cerrar()` or by leaving the context manager; leaking them would keep
    every forward pass of the model storing activations forever.
    """

    def __init__(self, modelo: nn.Module, capa: Optional[nn.Module] = None) -> None:
        self.modelo = modelo
        self.capa = capa if capa is not None else modelo.capa_objetivo
        self._activaciones: Optional[torch.Tensor] = None
        self._gradientes: Optional[torch.Tensor] = None
        self._manijas = [
            self.capa.register_forward_hook(self._guardar_activaciones),
            self.capa.register_full_backward_hook(self._guardar_gradientes),
        ]

    def _guardar_activaciones(self, _modulo, _entrada, salida) -> None:
        self._activaciones = salida.detach()

    def _guardar_gradientes(self, _modulo, _grad_entrada, grad_salida) -> None:
        self._gradientes = grad_salida[0].detach()

    def generar(
        self, tensor: torch.Tensor, indice_etiqueta: int, grupo: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Map for one label on one study.

        `tensor` is a single study, (G, C, H, W). When `grupo` is None the
        group the attention pooling weighted highest is used, which is the
        slice the model itself says it relied on.
        """
        if tensor.dim() != 4:
            raise ValueError("se esperaba un solo estudio: (grupos, canales, alto, ancho)")

        self.modelo.eval()
        self.modelo.zero_grad(set_to_none=True)

        entrada = tensor.unsqueeze(0).requires_grad_(True)
        salida = self.modelo(entrada)
        atencion = salida.atencion[0].detach().cpu().numpy()
        elegido = int(np.argmax(atencion)) if grupo is None else grupo

        logit = salida.logits[0, indice_etiqueta]
        probabilidad = float(torch.sigmoid(logit))
        logit.backward()

        mapa = self._combinar(elegido, tensor.shape[-2:])
        return mapa, atencion, probabilidad

    def _combinar(self, grupo: int, tamano: tuple[int, int]) -> np.ndarray:
        if self._activaciones is None or self._gradientes is None:
            raise RuntimeError("no se capturaron activaciones; revisá la capa objetivo")

        # The encoder saw (B*G) images, so row `grupo` is the group wanted.
        activaciones = self._activaciones[grupo]      # (K, h, w)
        gradientes = self._gradientes[grupo]          # (K, h, w)

        # Channel importance: how much raising this channel raises the logit.
        pesos = gradientes.mean(dim=(1, 2))           # (K,)
        combinado = torch.einsum("k,khw->hw", pesos, activaciones)

        # Only positive evidence: negative values argue against the label.
        combinado = F.relu(combinado)

        redimensionado = F.interpolate(
            combinado[None, None], size=tamano, mode="bilinear", align_corners=False
        )[0, 0]
        return _a_rango_unitario(redimensionado.cpu().numpy())

    def cerrar(self) -> None:
        for manija in self._manijas:
            manija.remove()
        self._manijas = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_) -> None:
        self.cerrar()


def _a_rango_unitario(mapa: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; an all-zero map stays all-zero rather than becoming noise."""
    maximo = float(mapa.max())
    if maximo <= 0:
        return np.zeros_like(mapa)
    return mapa / maximo


def superponer(imagen: np.ndarray, mapa: np.ndarray, alfa: float = 0.45) -> np.ndarray:
    """Blend a heat map over a grayscale slice, for the figures in the report."""
    import cv2

    gris = np.clip(imagen, 0.0, 1.0)
    fondo = cv2.cvtColor((gris * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    calor = cv2.applyColorMap((mapa * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(fondo, 1 - alfa, calor, alfa, 0)
