"""Dónde están los datos de la Pregunta 1.

**Todas las rutas quedan dentro de ``pregunta1/``.** No es una preferencia de
orden: esta carpeta se comparte por separado con el equipo, y cualquier ruta que
salga —``../data_kaggle/...``— no existiría en la máquina de quien la reciba, con
un error que no explica por qué.

``alinear_datos.py`` deja todo bajo ``pregunta1/datos/``:

| qué | tamaño | cómo |
|---|---|---|
| ``datos/raw/`` | 8.8 MB | **copia real**, sobrevive a cualquier forma de compartir |

Los DICOM no están acá porque **nada los lee desde disco**: el pipeline los saca
del ZIP y escribe tensores ``.npy`` en ``paso2_preprocesamiento/cache/``, que ya
vive dentro de la pregunta.

Lo único que queda afuera es el **ZIP del concurso (246.8 GB)**, que no es un dato
de esta pregunta sino la fuente de la que se extraen los DICOM. Se configura con
la variable de entorno ``RSNA_ZIP``; ver ``ruta_zip()``.

Usage:
    from pregunta1.rutas import TRAIN_CSV, SERIES_CSV, ruta_zip
"""

from __future__ import annotations

import os
import pathlib

#: La carpeta de esta pregunta, sin importar desde dónde se invoque el script.
AQUI = pathlib.Path(__file__).resolve().parent

#: Todo lo que esta pregunta lee vive acá adentro. **No hay ruta de respaldo
#: fuera del directorio**: si falta algo, el mensaje dice qué correr, en vez de
#: ir a buscarlo a una carpeta que en otra máquina no existe.
DATOS = AQUI / "datos"


def datos(relativa: str) -> pathlib.Path:
    """Resuelve una ruta bajo ``pregunta1/datos/``.

    ``relativa`` es la ruta tal como queda ahí adentro, por ejemplo
    ``"raw/train.csv"``.
    """
    return DATOS / relativa


TRAIN_CSV = datos("raw/train.csv")
SERIES_CSV = datos("raw/train_series.csv")


def ruta_zip() -> pathlib.Path:
    """El ZIP del concurso, que **no** vive dentro de la pregunta.

    Son 246.8 GB: es la fuente de la que se extraen los DICOM, no un dato que se
    comparta con la carpeta. Antes estaba escrito a mano como
    ``D:\\compartido\\rsna-knee-abnormality-detection.zip``, que es la peor forma
    posible de referenciarlo —una ruta absoluta de una máquina concreta—.

    Orden: la variable ``RSNA_ZIP``, y si no está, la ubicación histórica, que
    sirve de default en esta máquina sin obligar a nadie más a tenerla.
    """
    if os.environ.get("RSNA_ZIP"):
        return pathlib.Path(os.environ["RSNA_ZIP"])
    return pathlib.Path(r"D:\compartido\rsna-knee-abnormality-detection.zip")


def verificar() -> list[str]:
    """Qué falta, si falta algo. Lista vacía = todo en su lugar."""
    problemas = []
    for nombre, ruta in (("raw/train.csv", TRAIN_CSV),
                         ("raw/train_series.csv", SERIES_CSV)):
        if not ruta.is_file():
            problemas.append(f"falta {nombre}")
    if problemas:
        problemas.append("corre: python alinear_datos.py")
    return problemas


def describir() -> str:
    faltan = verificar()
    return ("datos de la Pregunta 1 en pregunta1/datos/" if not faltan
            else "datos de la Pregunta 1 INCOMPLETOS: " + "; ".join(faltan))


if __name__ == "__main__":
    print(describir())
    for nombre, ruta in (("raw/train.csv", TRAIN_CSV),
                         ("raw/train_series.csv", SERIES_CSV)):
        if ruta.is_file():
            print(f"  {nombre:<22} OK   {ruta.stat().st_size/2**20:>8.1f} MB")
        elif ruta.is_dir():
            n = sum(1 for _ in ruta.rglob("*") if _.is_file())
            print(f"  {nombre:<22} OK   {n:>8,} archivos")
        else:
            print(f"  {nombre:<22} FALTA")
    z = ruta_zip()
    print(f"\n  ZIP del concurso (fuera de la pregunta, 246 GB)")
    print(f"    {'OK' if z.is_file() else 'no accesible'}  {z}")
    print(f"    se cambia con la variable de entorno RSNA_ZIP")
