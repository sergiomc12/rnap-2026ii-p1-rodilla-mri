"""Turning one DICOM series into the tensor the models consume.

The output is 2.5D: N groups of three adjacent slices, each group stacked
as the three channels of one image. Three and not five or seven because
ImageNet weights expect exactly three channels, so a pretrained first
convolution loads unchanged -- which is what keeps the from-scratch model
and the transfer-learning model comparable on equal terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .geometria import (
    BANDA_CORTES, CROP_MM, LADO_SALIDA, MetadatosCorte, banda_central,
    es_rodilla_izquierda, normalizar_intensidad, ordenar_por_posicion,
    recortar_milimetros,
)

CORTES_POR_GRUPO = 3
GRUPOS_POR_SERIE = 8


@dataclass(frozen=True)
class ConfigPreproceso:
    """Every knob of the pipeline, frozen so a run cannot change it."""

    crop_mm: float = CROP_MM
    lado: int = LADO_SALIDA
    banda: tuple[float, float] = BANDA_CORTES
    cortes_por_grupo: int = CORTES_POR_GRUPO
    grupos: int = GRUPOS_POR_SERIE
    normalizar_lateralidad: bool = True
    #: El plano de la serie. NO es decorativo: decide si la normalizacion de
    #: lateralidad aplica. Invertir el orden de los cortes corrige que en una
    #: rodilla izquierda el barrido SAGITAL va de lateral a medial y en una
    #: derecha al reves --el eje entre cortes es medial-lateral y ese si se
    #: invierte con el lado--. En CORONAL el eje entre cortes es
    #: anterior-posterior y en AXIAL superior-inferior: ninguno de los dos
    #: depende de que rodilla sea, asi que invertirlos pondria los cortes
    #: posteriores donde van los anteriores. Medido: el 28% de los estudios
    #: son rodilla izquierda, o sea que aplicarlo en coronal corrompia uno de
    #: cada cuatro.
    plano: str = "Sagittal"

    def __post_init__(self) -> None:
        if self.crop_mm <= 0:
            raise ValueError("crop_mm must be > 0")
        if self.lado < 32:
            raise ValueError("lado must be >= 32")
        if self.cortes_por_grupo < 1:
            raise ValueError("cortes_por_grupo must be >= 1")
        if self.grupos < 1:
            raise ValueError("grupos must be >= 1")
        if not 0.0 <= self.banda[0] < self.banda[1] <= 1.0:
            raise ValueError("banda must be an increasing pair inside [0, 1]")

    @property
    def forma(self) -> tuple[int, int, int, int]:
        """Shape of one preprocessed series: (grupos, canales, alto, ancho)."""
        return (self.grupos, self.cortes_por_grupo, self.lado, self.lado)


def redimensionar(imagen: np.ndarray, lado: int) -> np.ndarray:
    """Resample to a square of `lado` pixels.

    Uses OpenCV's area interpolation when shrinking, which averages the
    pixels being merged instead of sampling one of them -- it matters here
    because most acquisitions are larger than the output.
    """
    import cv2

    if imagen.shape[0] == lado and imagen.shape[1] == lado:
        return imagen
    achicando = imagen.shape[0] > lado
    interpolacion = cv2.INTER_AREA if achicando else cv2.INTER_CUBIC
    return cv2.resize(imagen, (lado, lado), interpolation=interpolacion)


def indices_de_grupos(
    disponibles: Sequence[int], grupos: int, por_grupo: int
) -> list[list[int]]:
    """Pick `grupos` evenly spaced windows of `por_grupo` contiguous slices.

    Spreading the windows across the band covers the whole joint; taking
    them contiguous is what makes each group 2.5D instead of three
    unrelated pictures.
    """
    if len(disponibles) < por_grupo:
        # Too short a series: repeat the last slice rather than drop the study.
        relleno = list(disponibles) + [disponibles[-1]] * (por_grupo - len(disponibles))
        return [relleno] * grupos

    ultimo_inicio = len(disponibles) - por_grupo
    if grupos == 1:
        inicios = [ultimo_inicio // 2]
    else:
        paso = ultimo_inicio / (grupos - 1) if ultimo_inicio else 0.0
        inicios = [int(round(i * paso)) for i in range(grupos)]
    return [list(disponibles[i:i + por_grupo]) for i in inicios]


def preprocesar_serie(
    imagenes: Sequence[np.ndarray],
    metadatos: Sequence[MetadatosCorte],
    config: Optional[ConfigPreproceso] = None,
) -> np.ndarray:
    """Run the whole pipeline over one series.

    Returns an array of shape `config.forma`, float32 in [0, 1].
    """
    config = config or ConfigPreproceso()
    if not imagenes:
        raise ValueError("la serie no tiene cortes")

    orden = ordenar_por_posicion(range(len(imagenes)), metadatos)
    if (config.normalizar_lateralidad and config.plano == "Sagittal"
            and es_rodilla_izquierda(metadatos)):
        # Sagittal slices run medial-to-lateral on one knee and
        # lateral-to-medial on the other; reversing one side makes "the
        # third slice" mean the same anatomy in every study.
        #
        # Solo en sagital: ver la nota de `plano` en ConfigPreproceso.
        orden = list(reversed(orden))

    centrales = orden[banda_central(len(orden), config.banda)]
    grupos = indices_de_grupos(centrales, config.grupos, config.cortes_por_grupo)

    salida = np.zeros(config.forma, dtype=np.float32)
    for g, grupo in enumerate(grupos):
        for c, indice in enumerate(grupo):
            salida[g, c] = _preparar_corte(imagenes[indice], metadatos[indice], config)
    return salida


def _preparar_corte(
    imagen: np.ndarray, metadatos: MetadatosCorte, config: ConfigPreproceso
) -> np.ndarray:
    recortada = recortar_milimetros(imagen, metadatos.espaciado_mm, config.crop_mm)
    return redimensionar(normalizar_intensidad(recortada), config.lado)
