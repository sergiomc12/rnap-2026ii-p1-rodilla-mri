"""Preprocess the selected studies and cache them as .npy tensors.

Decoding and resampling is the slow part, so it runs once. Training then
reads tensors straight from disk instead of touching the 570 GB archive.

Usage:
    python pregunta1/paso2_preprocesamiento/generar_cache.py
    python pregunta1/paso2_preprocesamiento/generar_cache.py --limite 20
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta1.paso2_preprocesamiento.dicom import (  # noqa: E402
    leer_dicom, metadatos_de, pixeles_de,
)
from pregunta1.paso2_preprocesamiento.lector_zip import LectorZip  # noqa: E402
from pregunta1.paso2_preprocesamiento.preprocesamiento import (  # noqa: E402
    CORTES_POR_GRUPO, GRUPOS_POR_SERIE, LADO_SALIDA, ConfigPreproceso,
    preprocesar_serie,
)

SELECCION = _RAIZ / "pregunta1/paso2_preprocesamiento/estudios_seleccionados.csv"
CACHE = _RAIZ / "pregunta1/paso2_preprocesamiento/cache"


def procesar_estudio(lector: LectorZip, serie: str, config: ConfigPreproceso):
    """Decode one series and return its tensor, or None if unusable."""
    cortes = lector.cortes_de(serie)
    if not cortes:
        return None

    imagenes, metadatos = [], []
    for corte in cortes:
        conjunto = leer_dicom(lector.leer(corte))
        pixeles = pixeles_de(conjunto)
        if pixeles is None:
            continue
        imagenes.append(pixeles)
        metadatos.append(metadatos_de(conjunto))

    if not imagenes:
        return None
    return preprocesar_serie(imagenes, metadatos, config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seleccion", type=pathlib.Path, default=SELECCION)
    parser.add_argument("--cache", type=pathlib.Path, default=CACHE)
    parser.add_argument("--limite", type=int)
    parser.add_argument("--rehacer", action="store_true",
                        help="reprocesar aunque el .npy ya exista")
    # El numero de grupos decide cuantos cortes distintos ve el modelo:
    # g grupos de c cortes contiguos cubren hasta g*c cortes de la banda
    # central. Se expone para poder ablacionarlo en vez de discutirlo.
    parser.add_argument("--grupos", type=int, default=GRUPOS_POR_SERIE)
    parser.add_argument("--cortes-por-grupo", type=int,
                        default=CORTES_POR_GRUPO)
    # El lado del recorte en pixeles. El recorte fisico (crop_mm) no cambia:
    # se ve exactamente la misma region de la rodilla, con mas resolucion.
    # Los cuadernos del leaderboard usan 336-384; nosotros veniamos en 224.
    parser.add_argument("--lado", type=int, default=LADO_SALIDA)
    parser.add_argument("--plano", default="Sagittal",
                        choices=["Sagittal", "Coronal", "Axial"],
                        help="plano de las series del CSV. Decide si se aplica "
                             "la normalizacion de lateralidad, que solo vale "
                             "en sagital")
    args = parser.parse_args(argv)

    with args.seleccion.open(encoding="utf-8", newline="") as handle:
        filas = list(csv.DictReader(handle))
    if args.limite:
        filas = filas[:args.limite]

    # El plano NO es decorativo: decide si se aplica la inversion por
    # lateralidad, que corresponde en sagital y corrompe coronal y axial.
    # Omitirlo dejaba el 28% de los estudios coronales con los cortes
    # posteriores donde van los anteriores.
    config = ConfigPreproceso(grupos=args.grupos,
                              cortes_por_grupo=args.cortes_por_grupo,
                              lado=args.lado,
                              plano=args.plano)
    args.cache.mkdir(parents=True, exist_ok=True)
    print(f"{len(filas)} estudios -> {args.cache}  (forma {config.forma})\n")

    hechos = saltados = fallidos = 0
    comenzado = time.time()
    with LectorZip() as lector:
        for indice, fila in enumerate(filas, start=1):
            destino = args.cache / f"{fila['StudyInstanceUID']}.npy"
            if destino.is_file() and not args.rehacer:
                saltados += 1
                continue
            tensor = procesar_estudio(lector, fila["SeriesInstanceUID"], config)
            if tensor is None:
                fallidos += 1
                continue
            np.save(destino, tensor)
            hechos += 1
            if indice % 50 == 0:
                transcurrido = time.time() - comenzado
                print(f"  {indice:>4}/{len(filas)} | {transcurrido/60:5.1f} min | "
                      f"ETA {(len(filas)-indice)*transcurrido/indice/60:5.1f} min",
                      flush=True)

    tamano = sum(f.stat().st_size for f in args.cache.glob("*.npy")) / 1e9
    print(f"\nprocesados={hechos} saltados={saltados} fallidos={fallidos} | "
          f"{tamano:.2f} GB en disco | {(time.time()-comenzado)/60:.1f} min")
    return 1 if fallidos else 0


if __name__ == "__main__":
    raise SystemExit(main())
