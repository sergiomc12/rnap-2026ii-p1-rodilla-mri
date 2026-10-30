"""Pick the stratified subset of studies and write it to CSV.

Usage:
    python pregunta1/paso2_preprocesamiento/seleccionar_estudios.py
    python pregunta1/paso2_preprocesamiento/seleccionar_estudios.py --objetivo 400
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Optional, Sequence

# pregunta1/ entra al sys.path para poder importar esta pregunta como
# paquete: los scripts se corren por ruta, no con -m. Las rutas de datos
# salen de pregunta1/rutas.py: siempre dentro de pregunta1/datos/.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

from pregunta1 import rutas as _rutas  # noqa: E402

from pregunta1.paso2_preprocesamiento.lector_zip import LectorZip  # noqa: E402
from pregunta1.paso2_preprocesamiento.muestreo import (  # noqa: E402
    LABELS, MuestreadorEstratificado, cargar_etiquetados,
)
from pregunta1.estudios import load_series  # noqa: E402

ETIQUETAS = _RAIZ / "pregunta1/paso1_etiquetas_debiles/etiquetas_debiles.csv"
SERIES_CSV = _rutas.SERIES_CSV
SALIDA = _RAIZ / "pregunta1/paso2_preprocesamiento/estudios_seleccionados.csv"
CAMPOS = ["StudyInstanceUID", "SeriesInstanceUID", "idioma", "es_dorado",
          "n_cortes"] + list(LABELS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objetivo", type=int, default=600)
    parser.add_argument("--cuota", type=int, default=40,
                        help="positivos buscados por etiqueta")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--etiquetas", type=pathlib.Path, default=ETIQUETAS)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--plano", default="Sagittal",
                        choices=["Sagittal", "Coronal", "Axial"],
                        help="plano de la serie, para la ablacion del entregable #4")
    args = parser.parse_args(argv)

    etiquetados = cargar_etiquetados(args.etiquetas)
    muestra = MuestreadorEstratificado(
        objetivo=args.objetivo, cuota_por_etiqueta=args.cuota, semilla=args.semilla
    ).muestrear(etiquetados)

    print(f"MUESTRA DE {len(muestra.estudios)} ESTUDIOS")
    print(f"  dorados           : {muestra.dorados}")
    print(f"  por cuota         : {muestra.por_cuota}")
    print(f"  al azar           : {muestra.al_azar}")
    print(f"  sanos verificados : {muestra.sanos_verificados}")
    print(f"  descartados ilegibles: {muestra.descartados_ilegibles:,}\n")

    print(f"{'etiqueta':18} {'positivos':>10} {'prevalencia':>12}")
    for etiqueta, n in sorted(muestra.prevalencia().items(), key=lambda kv: kv[1]):
        print(f"{etiqueta:18} {n:>10} {100*n/len(muestra.estudios):>11.1f}%")

    filas = _elegir_series(muestra, args.plano)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    # Archivo temporal + rename, por el mismo motivo que en el paso 1.
    parcial = args.salida.with_suffix(args.salida.suffix + ".parcial")
    with parcial.open("w", encoding="utf-8", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)
    parcial.replace(args.salida)
    print(f"\n{len(filas)} estudios con serie {args.plano} -> {args.salida}")
    return 0


def _elegir_series(muestra, plano: str = "Sagittal") -> list[dict]:
    """One fluid-sensitive, fat-suppressed series per study, in one plane.

    The plane is a parameter because deliverable #4 names "eleccion de
    secuencia/plano de MRI" as the first ablation factor; running the same
    pipeline on Coronal is what turns that into a measurement.
    """
    series = load_series(SERIES_CSV)
    elegidas = {}
    for serie in series:
        if (serie.plane == plano and serie.fluid_sensitive
                and serie.fat_suppression):
            elegidas.setdefault(serie.study_uid, serie.series_uid)

    with LectorZip() as lector:
        filas = []
        sin_serie = 0
        for estudio in muestra.estudios:
            serie = elegidas.get(estudio.uid)
            if serie is None:
                sin_serie += 1
                continue
            fila = {
                "StudyInstanceUID": estudio.uid,
                "SeriesInstanceUID": serie,
                "idioma": estudio.idioma,
                "es_dorado": int(estudio.es_dorado),
                "n_cortes": len(lector.cortes_de(serie)),
            }
            fila.update(estudio.etiquetas)
            filas.append(fila)

    if sin_serie:
        print(f"\nAVISO: {sin_serie} estudios sin serie {plano} fluid-sensitive")
    return filas


if __name__ == "__main__":
    raise SystemExit(main())
