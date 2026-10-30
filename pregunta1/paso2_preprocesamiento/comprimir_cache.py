"""Convierte un caché de tensores float32 a uint8, para que entre en RAM.

Los tensores salen de `preprocesar_serie` normalizados a [0, 1], así que
guardarlos en float32 gasta 4 bytes por píxel para un número entre cero y uno.
En uint8 ocupan la cuarta parte:

    64 grupos x 3 cortes x 224 x 224    float32  36.8 MB   4 407 -> 156 GB
                                        uint8     9.2 MB   4 407 ->  39 GB

Con 128 GB de RAM, 39 GB entran enteros y el entrenamiento deja de tocar el
disco después de la primera época. En float32 no entraban, y la alternativa
--cargar por fragmentos-- rompe el barajado: todos los lotes de un tramo
saldrían del mismo fragmento y los gradientes consecutivos quedarían
correlacionados.

El error de cuantización medido sobre un tensor real es 0.00196 como máximo
--medio paso de 1/255-- sobre una imagen cuya señal útil tiene mucha menos
precisión que eso.

Usage:
    python pregunta1/paso2_preprocesamiento/comprimir_cache.py \
        --origen .../completo_g64_c192 --destino .../completo_g64_c192_u8
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Optional, Sequence

import numpy as np

#: Los limites de la transformacion afin. El rango real medido sobre 200
#: tensores es [-0.1589, 1.2323]; estos dejan margen y son FIJOS a proposito:
#: si cada tensor usara su propio min/max, dos estudios con rangos distintos
#: quedarian en escalas distintas y el modelo veria intensidades incomparables.
#: Cambiarlos invalida un cache ya convertido --hay que regenerarlo--.
LIMITE_BAJO, LIMITE_ALTO = -0.35, 1.35


def descomprimir(q: np.ndarray) -> np.ndarray:
    """uint8 -> float32 en el rango original. Es la inversa exacta."""
    escala = (LIMITE_ALTO - LIMITE_BAJO) / 255.0
    return q.astype(np.float32) * escala + LIMITE_BAJO


def convertir(origen: pathlib.Path, destino: pathlib.Path) -> tuple:
    """Devuelve (convertidos, saltados, error_maximo)."""
    destino.mkdir(parents=True, exist_ok=True)
    archivos = sorted(origen.glob("*.npy"))
    convertidos = saltados = 0
    peor = 0.0
    comenzado = time.time()
    for i, f in enumerate(archivos, 1):
        salida = destino / f.name
        if salida.is_file():
            saltados += 1
            continue
        a = np.load(f)
        if a.dtype == np.uint8:
            saltados += 1
            continue
        # El rango NO es [0,1] como dice el docstring de `preprocesar_serie`:
        # medido sobre 200 tensores es [-0.1589, 1.2323]. El desborde viene de
        # la interpolacion INTER_AREA al reescalar a 224x224, que sobrepasa en
        # los bordes de alto contraste --afecta al 10% de los tensores y al
        # 1.3% de sus pixeles--. Recortar a [0,1] alteraria esos pixeles, asi
        # que se mapea el rango completo con una afin fija.
        if a.min() < LIMITE_BAJO or a.max() > LIMITE_ALTO:
            raise ValueError(f"{f.name} fuera de [{LIMITE_BAJO}, {LIMITE_ALTO}]: "
                             f"[{a.min():.4f}, {a.max():.4f}]")
        escala = (LIMITE_ALTO - LIMITE_BAJO) / 255.0
        q = np.clip(np.rint((a - LIMITE_BAJO) / escala), 0, 255).astype(np.uint8)
        if convertidos < 20:              # el error se mide en una muestra
            rec = q.astype(np.float32) * escala + LIMITE_BAJO
            peor = max(peor, float(np.abs(a - rec).max()))
        # Se escribe sobre un descriptor ABIERTO, no sobre una ruta: `np.save`
        # le agrega ".npy" a cualquier nombre que no lo tenga, asi que pasarle
        # "x.parcial" crea "x.parcial.npy" y el renombrado posterior falla
        # buscando un archivo que nunca existio.
        tmp = salida.with_name(salida.name + ".parcial")
        with tmp.open("wb") as fh:
            np.save(fh, q)
        tmp.replace(salida)
        convertidos += 1
        if i % 250 == 0:
            transcurrido = time.time() - comenzado
            ritmo = i / max(transcurrido, 1e-9)
            print(f"  {i}/{len(archivos)} | {transcurrido/60:.1f} min | "
                  f"ETA {(len(archivos)-i)/ritmo/60:.1f} min", flush=True)
    return convertidos, saltados, peor


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origen", type=pathlib.Path, required=True)
    parser.add_argument("--destino", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    if not args.origen.is_dir():
        print(f"no existe {args.origen}")
        return 1
    convertidos, saltados, peor = convertir(args.origen, args.destino)
    tam = sum(f.stat().st_size for f in args.destino.glob("*.npy"))
    print(f"\nconvertidos={convertidos} saltados={saltados} | "
          f"{tam / 2**30:.1f} GB en disco")
    print(f"error de cuantizacion maximo medido: {peor:.5f}  (1/255 = {1/255:.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
