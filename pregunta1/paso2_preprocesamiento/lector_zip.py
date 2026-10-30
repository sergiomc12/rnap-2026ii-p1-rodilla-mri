"""Random access to the competition ZIP, without extracting it.

The archive holds 819,640 entries and 569.76 GB uncompressed, but `zipfile`
reads any entry by name straight from the central directory. Measured at
348 MB/s, so pulling the 600 studies this project needs takes under a
minute and the 570 GB never land on disk.

Downloading the same subset through the Kaggle API took 6h25m for 1.39 GB
and failed on 63% of the files to rate limiting, which is why nothing here
touches the network.
"""

from __future__ import annotations

import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pregunta1 import rutas as _rutas  # noqa: E402

# El ZIP del concurso son 246.8 GB: no vive dentro de la pregunta, porque no es
# un dato que se comparta con la carpeta sino la fuente de la que se extraen los
# DICOM. Antes estaba escrito acá a mano como una ruta absoluta de una máquina
# concreta, que es la peor forma posible de referenciarlo: invisible y rota para
# quien reciba la carpeta. Ahora sale de RSNA_ZIP, con la ubicación histórica
# como default para no romper nada en esta máquina.
ZIP_POR_DEFECTO = _rutas.ruta_zip()
PREFIJO_TRAIN = "train_series/"
SUFIJO_DICOM = ".dcm"


@dataclass(frozen=True)
class EntradaDicom:
    """One DICOM inside the archive, located by study and series."""

    nombre: str
    estudio: str
    serie: str
    instancia: str

    @classmethod
    def desde_nombre(cls, nombre: str) -> Optional["EntradaDicom"]:
        """Parse `train_series/<study>/<series>/<instance>.dcm`."""
        if not nombre.endswith(SUFIJO_DICOM):
            return None
        partes = nombre.split("/")
        if len(partes) != 4:
            return None
        return cls(nombre=nombre, estudio=partes[1], serie=partes[2],
                   instancia=partes[3][:-len(SUFIJO_DICOM)])


class LectorZip:
    """Reads DICOM bytes out of the competition archive on demand.

    The index is built once per instance: scanning 819,640 names takes a
    couple of seconds, and every later lookup is a dictionary hit.
    """

    def __init__(self, ruta: Path = ZIP_POR_DEFECTO) -> None:
        self.ruta = Path(ruta)
        if not self.ruta.is_file():
            raise FileNotFoundError(
                f"No está el ZIP de la competencia en {self.ruta}"
            )
        self._archivo = zipfile.ZipFile(self.ruta)

    @cached_property
    def entradas(self) -> tuple[EntradaDicom, ...]:
        """Every training DICOM in the archive."""
        return tuple(
            entrada
            for nombre in self._archivo.namelist()
            if nombre.startswith(PREFIJO_TRAIN)
            for entrada in [EntradaDicom.desde_nombre(nombre)]
            if entrada is not None
        )

    @cached_property
    def por_serie(self) -> dict[str, tuple[EntradaDicom, ...]]:
        """Slices grouped by series, in archive order."""
        agrupadas: dict[str, list[EntradaDicom]] = defaultdict(list)
        for entrada in self.entradas:
            agrupadas[entrada.serie].append(entrada)
        return {serie: tuple(items) for serie, items in agrupadas.items()}

    @cached_property
    def series_por_estudio(self) -> dict[str, tuple[str, ...]]:
        agrupadas: dict[str, set[str]] = defaultdict(set)
        for entrada in self.entradas:
            agrupadas[entrada.estudio].add(entrada.serie)
        return {estudio: tuple(sorted(series))
                for estudio, series in agrupadas.items()}

    def cortes_de(self, serie: str) -> tuple[EntradaDicom, ...]:
        """Slices of one series, or empty when the series is not here."""
        return self.por_serie.get(serie, ())

    def leer(self, entrada: EntradaDicom) -> bytes:
        """Raw DICOM bytes of one slice."""
        return self._archivo.read(entrada.nombre)

    def leer_varios(self, entradas: Iterable[EntradaDicom]) -> list[bytes]:
        return [self.leer(entrada) for entrada in entradas]

    def existe_serie(self, serie: str) -> bool:
        return serie in self.por_serie

    def filtrar_estudios(self, estudios: Sequence[str]) -> list[str]:
        """Keep only the studies the archive actually contains."""
        presentes = self.series_por_estudio
        return [estudio for estudio in estudios if estudio in presentes]

    def cerrar(self) -> None:
        self._archivo.close()

    def __enter__(self) -> "LectorZip":
        return self

    def __exit__(self, *_) -> None:
        self.cerrar()
