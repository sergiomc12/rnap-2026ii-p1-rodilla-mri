"""Un estudio visto por sus tres planos a la vez.

Motivo, medido: el plano que mejor muestra un hallazgo depende del hallazgo.
Comparando la MISMA arquitectura sobre sagital y sobre coronal, seis etiquetas
ganan en uno y seis en el otro --`PF OA` y `MCL` prefieren coronal, `Medial OA`
y `Effusion` prefieren sagital-- mientras el AUC macro casi no se mueve. El
promedio esconde que la mitad mejora mucho y la otra mitad empeora mucho.

La salida es ``(192, 3, 224, 224)``: los 64 grupos de cada plano apilados en un
orden FIJO --sagital 0-63, coronal 64-127, axial 128-191--. El orden tiene que
ser fijo porque la atencion por etiqueta aprende a ponderar POSICIONES, y la
posicion 100 debe significar siempre lo mismo.

Asi cada etiqueta decide sola de que plano sacar su informacion, en vez de que
se lo asignemos a mano con una tabla medida sobre 573 estudios y etiquetas
debiles.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import numpy as np

# Absoluta y no relativa: los scripts de este paso se corren POR RUTA, no con
# `-m`, asi que no hay paquete padre y `from .datos import ...` falla con
# "attempted relative import with no known parent package". `entrenar.py` mete
# la raiz en sys.path antes de importar, por eso esta forma si resuelve.
from pregunta1.paso3_modelos.datos import Registro

#: El orden es parte del contrato con el modelo: cambiarlo invalida los pesos.
PLANOS = ("sagital", "coronal", "axial")


class DatasetMultiplano:
    """Sirve (tensor de 3 planos, etiquetas) desde tres cachés alineados."""

    def __init__(
        self,
        registros: Sequence[Registro],
        caches: dict,
        aumentar: bool = False,
        semilla: int = 42,
        grupos_por_plano: int = 64,
    ) -> None:
        faltan = [p for p in PLANOS if p not in caches]
        if faltan:
            raise ValueError(f"faltan cachés: {faltan}")
        self.caches = {p: pathlib.Path(caches[p]) for p in PLANOS}
        self.grupos = grupos_por_plano
        self.aumentar = aumentar
        self._azar = np.random.RandomState(semilla)
        # Solo entran los estudios presentes en LOS TRES. Uno al que le falte
        # un plano daría un tensor con un plano repetido o desalineado, y eso
        # no se nota hasta mirar el AUC.
        self.registros = [r for r in registros
                          if all((self.caches[p] / f"{r.uid}.npy").is_file()
                                 for p in PLANOS)]

    def __len__(self) -> int:
        return len(self.registros)

    @property
    def prevalencia(self) -> dict:
        """Fraccion de positivos por etiqueta. La usa la perdida ponderada.

        Se calcula sobre los registros que QUEDARON --los que tienen los tres
        planos-- y no sobre los de entrada: si alguno se descartara, la
        prevalencia del conjunto real seria otra.
        """
        from pregunta1.paso3_modelos.datos import LABELS

        if not self.registros:
            return {etiqueta: 0.0 for etiqueta in LABELS}
        total = len(self.registros)
        return {
            etiqueta: sum(r.etiquetas[i] for r in self.registros) / total
            for i, etiqueta in enumerate(LABELS)
        }

    def __getitem__(self, indice: int):
        import torch

        registro = self.registros[indice]
        partes = [np.load(self.caches[p] / f"{registro.uid}.npy") for p in PLANOS]
        tensor = np.concatenate(partes, axis=0)      # (192, 3, 224, 224)

        if self.aumentar and self._azar.rand() < 0.5:
            # El volteo horizontal SOLO sobre el tramo sagital. Dentro de un
            # corte coronal o axial el eje horizontal de la imagen ES el
            # medial-lateral, asi que voltear intercambiaria `Medial OA` con
            # `Lateral OA` y los dos meniscos.
            tensor = tensor.copy()
            tensor[:self.grupos] = tensor[:self.grupos, ..., ::-1]

        return (
            torch.from_numpy(np.ascontiguousarray(tensor)),
            torch.tensor(registro.etiquetas, dtype=torch.float32),
        )

    def tramo(self, plano: str) -> slice:
        """Que grupos corresponden a un plano, para leer la atencion."""
        i = PLANOS.index(plano)
        return slice(i * self.grupos, (i + 1) * self.grupos)
