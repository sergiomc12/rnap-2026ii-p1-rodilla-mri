"""Translate every non-English report to English and cache the result.

Translation is the slow part of the pipeline, so it runs once and writes a
CSV. Everything downstream -- the extractor, the evaluation, the label
generation -- reads that file instead of loading the model again.

Usage:
    python pregunta1/paso1_etiquetas_debiles/traducir_reportes.py
    python pregunta1/paso1_etiquetas_debiles/traducir_reportes.py --limite 50
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from typing import Optional, Sequence

# pregunta1/ entra al sys.path para poder importar esta pregunta como
# paquete: los scripts se corren por ruta, no con -m. Las rutas de datos
# salen de pregunta1/rutas.py: siempre dentro de pregunta1/datos/.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

from pregunta1 import rutas as _rutas  # noqa: E402

from pregunta1.paso1_etiquetas_debiles.traduccion import TraductorNLLB  # noqa: E402
from pregunta1.estudios import load_studies  # noqa: E402

TRAIN_CSV = _rutas.TRAIN_CSV
SALIDA = _RAIZ / "pregunta1/paso1_etiquetas_debiles/reportes_traducidos.csv"
CAMPOS = ["StudyInstanceUID", "idioma", "fue_traducido", "reporte_en"]


def detectar_idioma(texto: str) -> str:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(texto[:600])
    except Exception:
        return "?"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=pathlib.Path, default=TRAIN_CSV)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--limite", type=int, help="traducir solo los primeros N")
    parser.add_argument("--dispositivo", help="cuda o cpu")
    args = parser.parse_args(argv)

    estudios = load_studies(args.train_csv)
    if args.limite:
        estudios = estudios[:args.limite]

    print(f"detectando idioma de {len(estudios):,} reportes...", flush=True)
    idiomas = [detectar_idioma(e.report) for e in estudios]
    a_traducir = sum(1 for i in idiomas if i != "en")
    print(f"a traducir: {a_traducir:,} | ya en ingles: {len(estudios)-a_traducir:,}\n")

    print("cargando NLLB-200-distilled-600M...", flush=True)
    comenzado = time.time()
    traductor = TraductorNLLB(dispositivo=args.dispositivo)
    print(f"cargado en {time.time()-comenzado:.0f}s | dispositivo: {traductor.dispositivo}\n",
          flush=True)

    comenzado = time.time()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=CAMPOS)
        escritor.writeheader()
        for indice, (estudio, idioma) in enumerate(zip(estudios, idiomas), start=1):
            traduccion = traductor.traducir_reporte(estudio.report, idioma)
            escritor.writerow({
                "StudyInstanceUID": estudio.uid,
                "idioma": idioma,
                "fue_traducido": int(traduccion.fue_traducido),
                "reporte_en": traduccion.texto,
            })
            if indice % 200 == 0:
                transcurrido = time.time() - comenzado
                restante = (len(estudios) - indice) * transcurrido / indice / 60
                print(f"  {indice:>5,}/{len(estudios):,} | "
                      f"{transcurrido/60:5.1f} min | ETA {restante:5.1f} min", flush=True)

    parcial.replace(args.salida)   # atomico: el artefacto aparece solo si termino
    print(f"\n{len(estudios):,} reportes -> {args.salida} "
          f"({(time.time()-comenzado)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
