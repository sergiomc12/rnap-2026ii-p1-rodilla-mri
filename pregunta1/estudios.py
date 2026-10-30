"""Lectura de `train.csv` y `train_series.csv`: estudios, series y los dorados.

**Por qué este módulo vive acá y no en `src/rsna/`.** Antes era
``src/rsna/selection.py``, fuera de ``pregunta1/``. Seis scripts de esta pregunta
lo importaban con ``sys.path[:0] = [..., str(_RAIZ / "src")]``, de modo que
**compartir solo la carpeta ``pregunta1/`` dejaba todo el paso 1 sin arrancar**:
``ModuleNotFoundError: rsna``. Arrastraba además ``DataPaths``, cuya raíz por
defecto era ``data_kaggle/`` —otra ruta de afuera—.

Acá no hay default que salga del directorio: las rutas por defecto son las de
[`rutas.py`](rutas.py), que resuelven dentro de ``pregunta1/datos/``.

Usage:
    from pregunta1.estudios import LABELS, gold_studies, load_studies
    oro = gold_studies(load_studies())
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass
from typing import Optional, Sequence

from . import rutas

#: Las doce etiquetas del concurso, en el orden en que las lista el enunciado.
#: Se repiten en ``paso1_etiquetas_debiles/terminos.py`` y en
#: ``paso3_modelos/datos.py``; un ``assert`` en cada uno verifica que coincidan.
LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)


@dataclass(frozen=True)
class Study:
    """Una fila de ``train.csv``: el informe y las doce etiquetas si están."""

    uid: str
    report: str
    labels: dict[str, int]

    @property
    def has_gold_labels(self) -> bool:
        """Verdadero para los 58 estudios que anotó un radiólogo.

        El criterio es tener **las doce** etiquetas, no alguna: en ``train.csv``
        las otras 4 349 filas las traen todas vacías.
        """
        return len(self.labels) == len(LABELS)

    @property
    def positives(self) -> frozenset[str]:
        return frozenset(n for n, v in self.labels.items() if v == 1)


@dataclass(frozen=True)
class Series:
    """Una fila de ``train_series.csv``: una adquisición de resonancia."""

    study_uid: str
    series_uid: str
    fluid_sensitive: bool
    fat_suppression: bool
    plane: str

    @property
    def is_sagittal_fluid_sensitive(self) -> bool:
        """La combinación a la que el enunciado recomienda restringirse."""
        return (self.plane == "Sagittal" and self.fluid_sensitive
                and self.fat_suppression)


def load_studies(path: Optional[pathlib.Path] = None) -> list[Study]:
    with (path or rutas.TRAIN_CSV).open(encoding="utf-8", newline="") as h:
        return [_estudio(f) for f in csv.DictReader(h)]


def load_series(path: Optional[pathlib.Path] = None) -> list[Series]:
    with (path or rutas.SERIES_CSV).open(encoding="utf-8", newline="") as h:
        return [_serie(f) for f in csv.DictReader(h)]


def gold_studies(estudios: Sequence[Study]) -> list[Study]:
    """El conjunto de validación limpio: los que tienen las doce etiquetas."""
    return [e for e in estudios if e.has_gold_labels]


def _estudio(fila: dict[str, str]) -> Study:
    etiquetas = {n: int(fila[n]) for n in LABELS
                 if fila.get(n, "").strip() != ""}
    return Study(uid=fila["StudyInstanceUID"], report=fila["Report"],
                 labels=etiquetas)


def _serie(fila: dict[str, str]) -> Series:
    return Series(study_uid=fila["StudyInstanceUID"],
                  series_uid=fila["SeriesInstanceUID"],
                  fluid_sensitive=fila["Fluid_Sensitive"] == "1",
                  fat_suppression=fila["Fat_Suppression"] == "1",
                  plane=fila["Anatomical_Plane"])
