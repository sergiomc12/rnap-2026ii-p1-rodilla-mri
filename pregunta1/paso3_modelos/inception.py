"""Inception encoder written from scratch, for deliverable #2.

Derived from GoogLeNet (`doc/GooGleNet.pdf`) and from the course lab
`notebooks/Lab_VGGNet_GoogLeNet.ipynb`, which already implements
`InceptionNaive`, `InceptionReduced` and `AuxClassifier`. Nothing here is
copied from a public Kaggle notebook.

Inception was chosen over a plain residual stack for a reason that matters
with this dataset: its parallel branches see 1x1, 3x3 and 5x5 receptive
fields at the same depth. Knee findings live at very different scales -- a
meniscal tear is a few millimetres, a joint effusion spans the whole
capsule -- and with only 576 studies, capturing both without going deeper
is worth more than raw depth.

The 1x1 reductions before the 3x3 and 5x5 branches are the point of the
"reduced" variant: without them the parameter count explodes, which is
exactly what a 576-study training set cannot afford.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _conv(entrada: int, salida: int, kernel: int, padding: int = 0) -> nn.Sequential:
    """Conv + BatchNorm + ReLU, the unit every block is built from."""
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
    )


@dataclass(frozen=True)
class AnchosInception:
    """Output channels of each branch of one Inception block."""

    rama1x1: int
    reduce3x3: int
    rama3x3: int
    reduce5x5: int
    rama5x5: int
    proyeccion_pool: int

    @property
    def salida(self) -> int:
        return self.rama1x1 + self.rama3x3 + self.rama5x5 + self.proyeccion_pool


class BloqueInception(nn.Module):
    """Four parallel branches concatenated along the channel axis."""

    def __init__(self, entrada: int, anchos: AnchosInception,
                 residual: bool = False) -> None:
        super().__init__()
        self.rama_1x1 = _conv(entrada, anchos.rama1x1, 1)
        self.rama_3x3 = nn.Sequential(
            _conv(entrada, anchos.reduce3x3, 1),          # 1x1 reduction first
            _conv(anchos.reduce3x3, anchos.rama3x3, 3, padding=1),
        )
        self.rama_5x5 = nn.Sequential(
            _conv(entrada, anchos.reduce5x5, 1),
            _conv(anchos.reduce5x5, anchos.rama5x5, 5, padding=2),
        )
        self.rama_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            _conv(entrada, anchos.proyeccion_pool, 1),
        )
        self.salida = anchos.salida

        # Conexion residual "de adelante hacia atras": la entrada del bloque se
        # suma a su salida (estilo Inception-ResNet). Las cuatro ramas ya
        # preservan alto y ancho (padding), asi que lo unico por igualar son los
        # canales -- de ahi la proyeccion 1x1, SIN ReLU, porque es un atajo
        # lineal, no una quinta rama. Queda OPCIONAL y apagada por defecto: con
        # `residual=False` este bloque es byte a byte el de antes, asi que
        # `inception` e `inception_googlenet` no cambian. Solo la encienden las
        # variantes *_residual de `construir_modelo`.
        #
        # El porque de probarlo: el comentario de CONFIGURACION_GOOGLENET midio
        # que nueve bloques SOBREAJUSTAN con 573 estudios (train baja, val sube).
        # El residual existe justamente para dejar entrenar una red mas profunda
        # sin ese sobreajuste -- es la hipotesis que esta variante pone a prueba.
        self.residual = residual
        if residual:
            self.proyeccion = (
                nn.Identity() if entrada == self.salida
                else nn.Sequential(
                    nn.Conv2d(entrada, self.salida, 1, bias=False),
                    nn.BatchNorm2d(self.salida),
                )
            )
            self.activacion = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concatenado = torch.cat([
            self.rama_1x1(x), self.rama_3x3(x), self.rama_5x5(x), self.rama_pool(x),
        ], dim=1)
        if not self.residual:
            return concatenado
        return self.activacion(concatenado + self.proyeccion(x))


# Four blocks, widening as the map shrinks. Smaller than the original
# GoogLeNet on purpose, and MEDIDO (10-oct-2026) en vez de supuesto: la
# version de nueve bloques (`CONFIGURACION_GOOGLENET`, 6.26 M de parametros)
# se entreno con el mismo protocolo sobre los dos conjuntos y hay un cruce.
#
#                         414 estudios de train   2 688 estudios
#   4 bloques (0.70 M)          0.628                 0.669
#   9 bloques (6.26 M)          0.607                 0.735
#   delta (9 - 4)          -0.021, no significativa  +0.065, SIGNIFICATIVA
#                                                     IC95% [+0.028, +0.105]
#
# Con 414 estudios los nueve bloques sobreajustan como decia la suposicion
# original --la perdida de entrenamiento baja de 0.428 a 0.372 mientras la de
# validacion sube de 0.703 a 1.006-- asi que para el entregable de la
# Pregunta 1, que usa 573 estudios por recomendacion del enunciado, cuatro
# bloques sigue siendo la eleccion correcta.
#
# Pero la justificacion NO se generaliza: con 2 688 estudios la relacion se
# invierte y los nueve bloques ganan por +0.065 con el intervalo entero por
# encima de cero. Si alguna vez se entrena sobre el conjunto completo, esta
# configuracion deja de ser la adecuada.
CONFIGURACION = (
    AnchosInception(32, 32, 64, 8, 16, 16),     # -> 128
    AnchosInception(64, 48, 96, 12, 32, 32),    # -> 224
    AnchosInception(96, 64, 128, 16, 48, 48),   # -> 320
    AnchosInception(128, 80, 192, 20, 64, 64),  # -> 448
)


# Los nueve modulos de GoogLeNet, con los anchos exactos de la Tabla 1 del
# paper (`doc/GooGleNet.pdf`), para poder MEDIR la afirmacion de arriba en vez
# de dejarla como suposicion: el comentario dice que nueve bloques
# sobreajustarian, y eso nunca se habia comprobado.
#
# Cada tupla es (#1x1, #3x3 reduce, #3x3, #5x5 reduce, #5x5, pool proj).
CONFIGURACION_GOOGLENET = (
    AnchosInception(64, 96, 128, 16, 32, 32),      # 3a -> 256
    AnchosInception(128, 128, 192, 32, 96, 64),    # 3b -> 480
    AnchosInception(192, 96, 208, 16, 48, 64),     # 4a -> 512
    AnchosInception(160, 112, 224, 24, 64, 64),    # 4b -> 512
    AnchosInception(128, 128, 256, 24, 64, 64),    # 4c -> 512
    AnchosInception(112, 144, 288, 32, 64, 64),    # 4d -> 528
    AnchosInception(256, 160, 320, 32, 128, 128),  # 4e -> 832
    AnchosInception(256, 160, 320, 32, 128, 128),  # 5a -> 832
    AnchosInception(384, 192, 384, 48, 128, 128),  # 5b -> 1024
)

#: Donde van los maxpool en GoogLeNet: despues de 3b (indice 1) y de 4e (6).
#: No es cada dos bloques: la etapa 4 tiene cinco modulos seguidos.
POOL_GOOGLENET = (1, 6)

#: El tronco de GoogLeNet termina en 192 canales, no en 96.
TALLO_GOOGLENET = 192


class EncoderInception(nn.Module):
    """Slice encoder: an image in, a feature vector out."""

    def __init__(self, canales_entrada: int = 3, configuracion=None,
                 pool_despues=None, canales_tallo: int = 96,
                 residual: bool = False) -> None:
        super().__init__()
        configuracion = configuracion or CONFIGURACION
        self.tallo = nn.Sequential(
            _conv(canales_entrada, 64, 7, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            _conv(64, 64, 1),
            _conv(64, canales_tallo, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        bloques, entrada = [], canales_tallo
        for indice, anchos in enumerate(configuracion):
            bloques.append(BloqueInception(entrada, anchos, residual=residual))
            entrada = anchos.salida
            # Sin `pool_despues` se cae al patron original: cada dos bloques.
            toca = (indice in pool_despues if pool_despues is not None
                    else indice % 2 == 1)
            if toca:
                bloques.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.bloques = nn.Sequential(*bloques)

        self.agrupado = nn.AdaptiveAvgPool2d(1)
        self.dimension = entrada

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bloques(self.tallo(x))
        return torch.flatten(self.agrupado(x), 1)

    @property
    def capa_objetivo(self) -> nn.Module:
        """Last convolutional stage, where Grad-CAM hooks in."""
        for modulo in reversed(list(self.bloques)):
            if isinstance(modulo, BloqueInception):
                return modulo
        raise RuntimeError("no hay bloques Inception")
