"""Putting every study into the same physical frame of reference.

Measured over 764 real slices, the acquisitions disagree on almost
everything: ten distinct matrix sizes from 320x320 to 1024x1024,
`PixelSpacing` from 0.146 to 0.562 mm/px (a factor of four), physical
fields of view from 130 to 240 mm, eight manufacturers, 1.5T and 3T, and
12- or 16-bit storage.

Resizing all of that straight to 224x224 would hand the network the same
meniscus at wildly different scales, so it would have to learn scanner
settings before it could learn anatomy. Cropping a fixed number of
*millimetres* first -- using `PixelSpacing` -- removes that variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

CROP_MM = 130.0  # covers the joint across the measured 130-240 mm fields of view
LADO_SALIDA = 224
BANDA_CORTES = (0.2, 0.8)  # the outer 20% at each end is mostly air and shaft


@dataclass(frozen=True)
class MetadatosCorte:
    """The handful of DICOM tags the geometry depends on."""

    filas: int
    columnas: int
    espaciado_mm: float
    posicion: Optional[tuple[float, float, float]] = None
    orientacion: Optional[tuple[float, ...]] = None
    lateralidad: str = ""

    @property
    def campo_vision_mm(self) -> float:
        return self.filas * self.espaciado_mm


def recortar_milimetros(
    imagen: np.ndarray, espaciado_mm: float, crop_mm: float = CROP_MM
) -> np.ndarray:
    """Centre-crop a fixed physical size, in millimetres.

    A smaller acquisition than `crop_mm` is returned untouched rather than
    padded: padding would invent anatomy that was never scanned.
    """
    if espaciado_mm <= 0:
        return imagen
    lado_px = int(round(crop_mm / espaciado_mm))
    alto, ancho = imagen.shape[:2]
    if lado_px >= min(alto, ancho):
        return imagen

    arriba = (alto - lado_px) // 2
    izquierda = (ancho - lado_px) // 2
    return imagen[arriba:arriba + lado_px, izquierda:izquierda + lado_px]


def normalizar_intensidad(imagen: np.ndarray, percentiles: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Scale one slice to [0, 1] using its own percentiles.

    MRI has no absolute units -- the same tissue reads 636 on one scanner
    and 2781 on another -- so normalization is per slice, and clipping at
    the 1st/99th percentile keeps a single bright artifact from flattening
    everything else.
    """
    datos = imagen.astype(np.float32)
    bajo, alto = np.percentile(datos, percentiles)
    if alto <= bajo:
        return np.zeros_like(datos)
    return np.clip((datos - bajo) / (alto - bajo), 0.0, 1.0)


def banda_central(cantidad: int, banda: tuple[float, float] = BANDA_CORTES) -> slice:
    """Index range covering the middle fraction of a series."""
    inicio = int(cantidad * banda[0])
    fin = max(inicio + 1, int(cantidad * banda[1]))
    return slice(inicio, fin)


def es_rodilla_izquierda(metadatos: Sequence[MetadatosCorte]) -> Optional[bool]:
    """Decide which knee this is, or give up honestly.

    The `Laterality` tag is missing or blank on 54% of the measured slices,
    so it is only the first source. The fallback uses the patient
    coordinate system, where +x points to the patient's left: a knee
    scanned on the left side sits at positive x.

    Returns None when neither source answers, so the caller can leave the
    series alone instead of flipping it on a guess.
    """
    for corte in metadatos:
        marca = (corte.lateralidad or "").strip().upper()
        if marca in ("L", "LEFT"):
            return True
        if marca in ("R", "RIGHT"):
            return False

    posiciones = [c.posicion[0] for c in metadatos if c.posicion is not None]
    if not posiciones:
        return None
    return float(np.mean(posiciones)) > 0


def ordenar_por_posicion(
    indices: Sequence[int], metadatos: Sequence[MetadatosCorte]
) -> list[int]:
    """Sort slices along the acquisition axis.

    Archive order is alphabetical by instance UID, which has nothing to do
    with anatomy. Sagittal slices must run medial-to-lateral for the stack
    to mean the same thing across studies.
    """
    def clave(indice: int) -> float:
        posicion = metadatos[indice].posicion
        return posicion[0] if posicion is not None else float(indice)

    return sorted(indices, key=clave)
