"""Reading DICOM bytes into pixels and the tags geometry needs."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

from .geometria import MetadatosCorte


def leer_dicom(datos: bytes):
    """Parse one DICOM from raw bytes."""
    import pydicom

    return pydicom.dcmread(io.BytesIO(datos))


def metadatos_de(conjunto) -> MetadatosCorte:
    """Pull the geometry tags, tolerating the ones that are often absent.

    `Laterality` is blank or missing on 54% of the measured slices and
    `ImagePositionPatient` is not guaranteed either, so both are optional
    and the caller decides what to do without them.
    """
    espaciado = getattr(conjunto, "PixelSpacing", None)
    posicion = getattr(conjunto, "ImagePositionPatient", None)
    orientacion = getattr(conjunto, "ImageOrientationPatient", None)
    return MetadatosCorte(
        filas=int(getattr(conjunto, "Rows", 0)),
        columnas=int(getattr(conjunto, "Columns", 0)),
        espaciado_mm=float(espaciado[0]) if espaciado else 0.0,
        posicion=tuple(float(v) for v in posicion) if posicion else None,
        orientacion=tuple(float(v) for v in orientacion) if orientacion else None,
        lateralidad=str(getattr(conjunto, "Laterality", "") or ""),
    )


def pixeles_de(conjunto) -> Optional[np.ndarray]:
    """Decode the pixel data, or None when the transfer syntax fails.

    The competition mixes uncompressed Explicit VR Little Endian with JPEG
    Lossless, JPEG 2000 and Implicit VR. The compressed ones need
    `pylibjpeg` or `gdcm` installed; returning None lets a run skip one bad
    slice instead of losing the study.
    """
    try:
        return conjunto.pixel_array
    except Exception:
        return None
