"""The two multi-label architectures deliverable #2 asks for.

Both share everything except the encoder: the same 2.5D input, the same
attention pooling, the same twelve-sigmoid head, the same training loop.
That is deliberate -- deliverable #3 asks to "comparar su rendimiento de
forma rigurosa", and a comparison is only rigorous when one thing varies.

  A. `EncoderInception`  -- written from scratch (see inception.py)
  B. `EncoderResNet50`   -- ImageNet weights, which the exam names explicitly
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .inception import EncoderInception
from .pooling import SalidaPooling, construir_pooling

NUM_ETIQUETAS = 12


class EncoderResNet50(nn.Module):
    """ResNet-50 with ImageNet weights, used as a slice feature extractor.

    The exam names this one: "un modelo preentrenado en ImageNet con
    fine-tuning, como ResNet-50 o EfficientNet". The classifier head is
    dropped; what is kept is the 2048-dimensional pooled feature.

    The input is three contiguous slices rather than RGB, which is why
    step 2 groups slices in threes: the pretrained first convolution loads
    unchanged.
    """

    def __init__(self, preentrenado: bool = True, congelar_hasta: int = 0) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        pesos = ResNet50_Weights.IMAGENET1K_V2 if preentrenado else None
        red = resnet50(weights=pesos)
        self.dimension = red.fc.in_features
        red.fc = nn.Identity()
        self.red = red
        self.preentrenado = preentrenado
        if congelar_hasta:
            self._congelar(congelar_hasta)

    def _congelar(self, cantidad: int) -> None:
        """Freeze the first `cantidad` top-level stages."""
        etapas = [self.red.conv1, self.red.bn1, self.red.layer1,
                  self.red.layer2, self.red.layer3, self.red.layer4]
        for etapa in etapas[:cantidad]:
            for parametro in etapa.parameters():
                parametro.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)

    @property
    def capa_objetivo(self) -> nn.Module:
        """Last convolutional stage, where Grad-CAM hooks in."""
        return self.red.layer4


@dataclass(frozen=True)
class SalidaModelo:
    """Logits plus the attention weights that produced them."""

    logits: torch.Tensor      # (B, 12)
    atencion: torch.Tensor    # (B, G)

    def probabilidades(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)


class ClasificadorRodilla(nn.Module):
    """Encoder over slice groups, attention pooling, twelve sigmoid outputs.

    Input is (B, G, C, H, W): B studies, G slice groups, C contiguous
    slices per group. The groups are folded into the batch so the encoder
    sees ordinary images, then unfolded for pooling.
    """

    def __init__(
        self,
        encoder: nn.Module,
        pooling: str = "atencion",
        num_etiquetas: int = NUM_ETIQUETAS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = construir_pooling(pooling, encoder.dimension,
                                         etiquetas=num_etiquetas)
        # Con pooling por etiqueta cada etiqueta recibe SU PROPIO vector de D,
        # asi que una Linear(D, 12) compartida ya no sirve: leeria el vector
        # equivocado. Hacen falta doce Linear(D, 1), que se implementan como un
        # peso (E, D) y un sesgo (E,) para resolverlas en un solo einsum.
        self.por_etiqueta = pooling.startswith("atencion_por_etiqueta")
        self.descarte = nn.Dropout(dropout)
        if self.por_etiqueta:
            self.peso = nn.Parameter(torch.empty(num_etiquetas, encoder.dimension))
            self.sesgo = nn.Parameter(torch.zeros(num_etiquetas))
            nn.init.normal_(self.peso, std=encoder.dimension ** -0.5)
            self.cabeza = None
        else:
            self.cabeza = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(encoder.dimension, num_etiquetas),
            )

    def forward(self, x: torch.Tensor) -> SalidaModelo:
        if x.dim() != 5:
            raise ValueError("se esperaba (lote, grupos, canales, alto, ancho)")
        lote, grupos = x.shape[:2]

        plano = x.flatten(0, 1)                       # (B*G, C, H, W)
        caracteristicas = self.encoder(plano)         # (B*G, D)
        por_grupo = caracteristicas.view(lote, grupos, -1)

        agrupado = self.pooling(por_grupo)
        if self.por_etiqueta:
            # embedding es (B, E, D): cada etiqueta lee SOLO su propio vector.
            # El einsum con "ed" en vez de una matriz plena es justamente eso:
            # la etiqueta e se multiplica por el vector e, nunca por los otros.
            emb = self.descarte(agrupado.embedding)
            logits = torch.einsum("bed,ed->be", emb, self.peso) + self.sesgo
        else:
            logits = self.cabeza(agrupado.embedding)
        return SalidaModelo(logits=logits, atencion=agrupado.atencion)

    @property
    def capa_objetivo(self) -> nn.Module:
        return self.encoder.capa_objetivo


def construir_modelo(
    nombre: str, pooling: str = "atencion", preentrenado: bool = True
) -> ClasificadorRodilla:
    """Build one of the two architectures by name.

    ``inception_googlenet`` NO es una tercera arquitectura del entregable: es la
    misma, con los nueve modulos y los anchos exactos de GoogLeNet en vez de los
    cuatro reducidos. Existe para medir si la version chica se eligio bien, en
    lugar de dejarlo como una afirmacion del comentario del codigo.
    """
    if nombre == "inception":
        encoder = EncoderInception()
    elif nombre == "inception_residual":
        # Misma Inception de 4 bloques, con el atajo residual encendido.
        encoder = EncoderInception(residual=True)
    elif nombre == "inception_googlenet":
        from .inception import (CONFIGURACION_GOOGLENET, POOL_GOOGLENET,
                                TALLO_GOOGLENET)
        encoder = EncoderInception(configuracion=CONFIGURACION_GOOGLENET,
                                   pool_despues=POOL_GOOGLENET,
                                   canales_tallo=TALLO_GOOGLENET)
    elif nombre == "inception_googlenet_residual":
        # La version PROFUNDA (9 modulos) CON residual: la combinacion que la
        # hipotesis dice que deberia entrenar sin el sobreajuste que la version
        # sin atajo sufre con 573 estudios.
        from .inception import (CONFIGURACION_GOOGLENET, POOL_GOOGLENET,
                                TALLO_GOOGLENET)
        encoder = EncoderInception(configuracion=CONFIGURACION_GOOGLENET,
                                   pool_despues=POOL_GOOGLENET,
                                   canales_tallo=TALLO_GOOGLENET,
                                   residual=True)
    elif nombre == "resnet50":
        encoder = EncoderResNet50(preentrenado=preentrenado)
    else:
        raise ValueError(f"modelo desconocido: {nombre}")
    return ClasificadorRodilla(encoder, pooling=pooling)
