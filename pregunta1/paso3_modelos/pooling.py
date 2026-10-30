"""Aggregating slice features into one prediction per study.

This is the part the exam requires the students to implement: "los
componentes principales de la arquitectura deben ser implementados por el
estudiante". The encoders may come from pretrained weights, but the step
from eight slice groups to twelve probabilities does not.

A mean over groups would dilute the finding: a meniscal tear spans two or
three slices out of thirty, so averaging drowns it in healthy tissue. A max
would pick a single group and throw away corroborating evidence, and it is
brittle against one noisy slice.

Attention pooling learns which groups to look at. It also leaves an audit
trail -- the weights say *which slice* the model used -- which pairs with
Grad-CAM saying *where inside that slice*, for deliverable #5.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SalidaPooling:
    """The pooled embedding and the weights that produced it."""

    embedding: torch.Tensor      # (B, D)
    atencion: torch.Tensor       # (B, G)


class AtencionPooling(nn.Module):
    """Gated attention over a variable number of slice groups.

    Follows the attention-based multiple instance learning formulation:
    a small network scores each group, the scores are softmaxed into
    weights that sum to one, and the study embedding is their weighted sum.

    The gate (the sigmoid branch) lets the network suppress a group whose
    content is irrelevant even if its score is high, which matters here
    because most slices of a knee are healthy tissue.
    """

    def __init__(self, dimension: int, oculta: int = 128, con_compuerta: bool = True) -> None:
        super().__init__()
        if dimension < 1 or oculta < 1:
            raise ValueError("dimension y oculta deben ser >= 1")
        self.con_compuerta = con_compuerta
        self.atencion = nn.Sequential(nn.Linear(dimension, oculta), nn.Tanh())
        self.compuerta = (nn.Sequential(nn.Linear(dimension, oculta), nn.Sigmoid())
                          if con_compuerta else None)
        self.puntaje = nn.Linear(oculta, 1)

    def forward(self, caracteristicas: torch.Tensor) -> SalidaPooling:
        """`caracteristicas` is (B, G, D); returns the study embedding."""
        if caracteristicas.dim() != 3:
            raise ValueError("se esperaba (lote, grupos, dimension)")

        oculto = self.atencion(caracteristicas)
        if self.compuerta is not None:
            oculto = oculto * self.compuerta(caracteristicas)

        puntajes = self.puntaje(oculto).squeeze(-1)          # (B, G)
        pesos = torch.softmax(puntajes, dim=1)
        embedding = torch.einsum("bg,bgd->bd", pesos, caracteristicas)
        return SalidaPooling(embedding=embedding, atencion=pesos)


class PromedioPooling(nn.Module):
    """Plain mean over groups, kept as the ablation baseline."""

    def forward(self, caracteristicas: torch.Tensor) -> SalidaPooling:
        if caracteristicas.dim() != 3:
            raise ValueError("se esperaba (lote, grupos, dimension)")
        grupos = caracteristicas.size(1)
        pesos = caracteristicas.new_full(caracteristicas.shape[:2], 1.0 / grupos)
        return SalidaPooling(embedding=caracteristicas.mean(dim=1), atencion=pesos)


@dataclass(frozen=True)
class SalidaPoolingPorEtiqueta:
    """Un embedding y una ponderacion POR ETIQUETA."""

    embedding: torch.Tensor      # (B, E, D)
    atencion: torch.Tensor       # (B, E, G)


class AtencionPorEtiqueta(nn.Module):
    """Atencion compartida hasta la proyeccion, separada desde la compuerta.

    El motivo es anatomico y se ve en los datos: los grupos que importan para
    ``Baker's`` son los posteriores, los de ``PF OA`` los anteriores, y el
    menisco medial y el lateral estan en extremos opuestos del barrido ---
    medido sobre los volumenes de SKM-TEA, 57 cortes de separacion en promedio
    y ningun volumen donde convivan. Con una sola ponderacion para las doce
    etiquetas, la red tiene que elegir un unico "donde mirar" que sirva para
    todo, y eso no existe.

    El corte esta en la compuerta, no en el puntaje, por un criterio explicito:
    se comparte todo lo que calcula QUE HAY en el corte --y eso no depende de
    la pregunta-- y se separa desde la primera operacion que DECIDE algo. La
    proyeccion ``atencion`` es una transformacion neutra; la compuerta ya
    suprime, asi que queda del lado separado.

    Compartir el backbone esta respaldado por una medicion de este proyecto:
    pesos de ImageNet --fotos de gatos y autos, ningun conocimiento de
    rodillas-- aportan +0.084 de AUC, significativo. Si rasgos de un dominio
    ajeno sirven tanto, los rasgos no son especificos de la etiqueta.

    Importante: el costo NO crece con la cantidad de grupos. El puntaje se
    aplica por grupo en paralelo, asi que pasar de 8 a 64 grupos no agrega un
    solo parametro --solo computo del backbone, que corre una vez por grupo--.
    """

    def __init__(self, dimension: int, etiquetas: int, oculta: int = 128,
                 con_compuerta: bool = True) -> None:
        super().__init__()
        if dimension < 1 or oculta < 1 or etiquetas < 1:
            raise ValueError("dimension, oculta y etiquetas deben ser >= 1")
        self.etiquetas = etiquetas
        self.con_compuerta = con_compuerta
        # Compartido: proyectar no decide nada.
        self.atencion = nn.Sequential(nn.Linear(dimension, oculta), nn.Tanh())
        # Por etiqueta: desde aca cada una decide lo suyo. Se implementan como
        # una sola capa ancha en vez de una ModuleList de doce, para que las
        # doce ramas corran en UNA multiplicacion de matrices y no en un bucle
        # de Python por etiqueta.
        self.compuerta = (nn.Linear(dimension, oculta * etiquetas)
                          if con_compuerta else None)
        self.puntaje = nn.Linear(oculta, etiquetas)

    def forward(self, caracteristicas: torch.Tensor) -> SalidaPoolingPorEtiqueta:
        """`caracteristicas` es (B, G, D); devuelve (B, E, D) y (B, E, G)."""
        if caracteristicas.dim() != 3:
            raise ValueError("se esperaba (lote, grupos, dimension)")
        lote, grupos, _ = caracteristicas.shape
        E, H = self.etiquetas, self.puntaje.in_features

        oculto = self.atencion(caracteristicas)              # (B, G, H)
        if self.compuerta is not None:
            puerta = self.compuerta(caracteristicas).view(lote, grupos, E, H)
            # (B,G,1,H) * (B,G,E,H) -> cada etiqueta suprime lo suyo
            oculto = oculto.unsqueeze(2) * torch.sigmoid(puerta)
        else:
            oculto = oculto.unsqueeze(2).expand(lote, grupos, E, H)

        # Un puntaje por (grupo, etiqueta): se toma la diagonal de la capa
        # ancha, que es el puntaje de cada etiqueta sobre SU propio oculto.
        puntajes = torch.einsum("bgeh,he->bge", oculto, self.puntaje.weight.t())
        puntajes = puntajes + self.puntaje.bias                # (B, G, E)
        pesos = torch.softmax(puntajes, dim=1)                 # sobre GRUPOS
        embedding = torch.einsum("bge,bgd->bed", pesos, caracteristicas)
        return SalidaPoolingPorEtiqueta(embedding=embedding,
                                        atencion=pesos.transpose(1, 2))


def construir_pooling(nombre: str, dimension: int,
                      etiquetas: int = 12) -> nn.Module:
    """Factory used by the ablation of deliverable #4."""
    if nombre == "atencion":
        return AtencionPooling(dimension)
    if nombre == "atencion_sin_compuerta":
        return AtencionPooling(dimension, con_compuerta=False)
    if nombre == "atencion_por_etiqueta":
        return AtencionPorEtiqueta(dimension, etiquetas)
    if nombre == "atencion_por_etiqueta_sin_compuerta":
        return AtencionPorEtiqueta(dimension, etiquetas, con_compuerta=False)
    if nombre == "promedio":
        return PromedioPooling()
    raise ValueError(f"pooling desconocido: {nombre}")
