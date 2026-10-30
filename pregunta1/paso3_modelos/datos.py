"""Reading the cached tensors as a PyTorch dataset.

Step 2 already did the expensive work -- decoding DICOM, cropping to a
fixed physical size, normalizing intensity -- so training only reads .npy
files. That is what makes an epoch cheap enough to run the ablations the
exam asks for.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)


@dataclass(frozen=True)
class Registro:
    """One study: where its tensor lives and what its labels say."""

    uid: str
    ruta: Path
    etiquetas: tuple[int, ...]
    es_dorado: bool
    idioma: str

    @property
    def existe(self) -> bool:
        return self.ruta.is_file()


def cargar_registros(
    seleccion: Path, cache: Path, etiquetas_reales: Optional[dict] = None
) -> list[Registro]:
    """Read the selection CSV and pair each study with its cached tensor.

    `etiquetas_reales` overrides the weak labels for the studies it covers;
    it is how the gold set gets its radiologist labels instead of the
    extractor's guess.
    """
    with Path(seleccion).open(encoding="utf-8", newline="") as handle:
        filas = list(csv.DictReader(handle))

    registros = []
    for fila in filas:
        uid = fila["StudyInstanceUID"]
        reales = (etiquetas_reales or {}).get(uid)
        etiquetas = (tuple(reales) if reales is not None
                     else tuple(int(fila[e]) for e in LABELS))
        registros.append(Registro(
            uid=uid,
            ruta=Path(cache) / f"{uid}.npy",
            etiquetas=etiquetas,
            es_dorado=fila["es_dorado"] == "1",
            idioma=fila["idioma"],
        ))
    return [r for r in registros if r.existe]


class DatasetRodilla:
    """Serves (tensor, labels) pairs from the cache.

    Augmentation is horizontal flip only. It is the one transform that is
    safe here: the knee has no left-right semantics inside a sagittal
    slice, while rotation or shear would distort the physical scale that
    step 2 went to the trouble of normalizing.

    **Y es seguro SOLO en sagital.** Dentro de un corte coronal el eje
    horizontal de la imagen ES el medial-lateral, asi que voltearlo
    intercambia `Medial OA` con `Lateral OA` y los dos meniscos: le ensenaria
    al modelo que el menisco medial puede estar de cualquier lado, que es lo
    contrario de lo que tiene que aprender. En axial pasa lo mismo. Por eso
    `voltear` se apaga para cualquier plano que no sea sagital; el mismo error
    ya aparecio en la herramienta de rodilla, donde corrompia la etiqueta de
    4 de las 6 clases la mitad de las veces.
    """

    def __init__(
        self,
        registros: Sequence[Registro],
        aumentar: bool = False,
        semilla: int = 42,
        en_memoria: bool = False,
        voltear: bool = True,
    ) -> None:
        self.registros = list(registros)
        self.aumentar = aumentar
        # El volteo horizontal SOLO es valido en sagital: ver el docstring.
        self.voltear = voltear
        self._azar = np.random.RandomState(semilla)
        self._memoria = None
        if en_memoria:
            self._memoria = self._precargar()

    def _precargar(self) -> list:
        """Sube todos los tensores a RAM una sola vez.

        Con 64 grupos cada tensor pesa 36.8 MB en float32 y una epoca lee
        156 GB. Medido: el disco da 1 738 MB/s y el entrenamiento lograba 235,
        porque la lectura y el computo se turnaban en el mismo proceso en vez
        de solaparse. Teniendolo todo en RAM el disco deja de intervenir.

        Solo tiene sentido con el cache en uint8 (9.2 MB por tensor, 39 GB el
        conjunto); en float32 serian 156 GB y no entra en los 128 de la maquina.
        """
        import sys

        total = 0
        datos = []
        for i, r in enumerate(self.registros, 1):
            a = np.load(r.ruta)
            datos.append(a)
            total += a.nbytes
            if i % 500 == 0:
                print(f"    precargando {i}/{len(self.registros)} "
                      f"({total / 2**30:.1f} GB)", flush=True, file=sys.stderr)
        print(f"    {len(datos)} tensores en RAM, {total / 2**30:.1f} GB "
              f"({datos[0].dtype})", flush=True, file=sys.stderr)
        return datos

    def __len__(self) -> int:
        return len(self.registros)

    def __getitem__(self, indice: int):
        import torch

        registro = self.registros[indice]
        tensor = (self._memoria[indice] if self._memoria is not None
                  else np.load(registro.ruta))
        if self.aumentar and self.voltear and self._azar.rand() < 0.5:
            tensor = tensor[..., ::-1].copy()
        t = torch.from_numpy(np.ascontiguousarray(tensor))
        if t.dtype == torch.uint8:
            # El cache uint8 guarda una afin del rango real --que NO es [0,1]:
            # la interpolacion al reescalar desborda hasta [-0.16, 1.23]-- y se
            # deshace aca, no en el preprocesamiento, para que el modelo reciba
            # exactamente los mismos numeros que con el cache float32.
            from ..paso2_preprocesamiento.comprimir_cache import (
                LIMITE_ALTO, LIMITE_BAJO)
            escala = (LIMITE_ALTO - LIMITE_BAJO) / 255.0
            t = t.float().mul_(escala).add_(LIMITE_BAJO)
        return (t, torch.tensor(registro.etiquetas, dtype=torch.float32))

    @property
    def prevalencia(self) -> dict[str, float]:
        if not self.registros:
            return {etiqueta: 0.0 for etiqueta in LABELS}
        total = len(self.registros)
        return {
            etiqueta: sum(r.etiquetas[i] for r in self.registros) / total
            for i, etiqueta in enumerate(LABELS)
        }
