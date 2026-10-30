"""Run the extractor over every training report and write the weak labels.

This is the artifact step 2 consumes: without it there is nothing to train
on and nothing to stratify the study subset by.

Usage:
    python pregunta1/paso1_etiquetas_debiles/generar_etiquetas.py
    python pregunta1/paso1_etiquetas_debiles/generar_etiquetas.py --salida otro.csv
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

from pregunta1.paso1_etiquetas_debiles import LABELS, ExtractorReglas  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.normalidad import clasificar  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.oraciones import dividir_en_oraciones  # noqa: E402
from pregunta1.estudios import Study, load_studies  # noqa: E402

TRAIN_CSV = _rutas.TRAIN_CSV
TRADUCIDOS = _RAIZ / "pregunta1/paso1_etiquetas_debiles/reportes_traducidos.csv"
SALIDA = _RAIZ / "pregunta1/paso1_etiquetas_debiles/etiquetas_debiles.csv"

# The extractor reads English and Spanish natively and nothing else, so only
# the other seven languages are replaced by their translation. Measured on the
# 58 gold studies: translating everything gives macro F1 0.551, translating
# nothing 0.512, and this hybrid 0.584 -- because NLLB degrades Spanish medical
# terminology (0.48 -> 0.29) while it rescues the languages that scored zero.
IDIOMAS_NATIVOS = frozenset({"en", "es"})

# Beyond the labels, step 2 needs to know how much to trust each row: which
# studies have a radiologist's verdict, which language the report was in
# (the extractor only covers English and Spanish), and how much evidence
# the extractor actually found. That last one feeds the noisy-label
# ablation of deliverable #4.
# ``veredicto`` separa dos cosas que producen el mismo vector de doce ceros:
# una rodilla sana y un informe que el extractor no supo leer. Tratarlas
# igual costaba los ejemplos negativos del entrenamiento (ver normalidad.py).
CAMPOS_META = ["StudyInstanceUID", "idioma", "traducido", "n_oraciones",
               "n_evidencias", "es_dorado", "veredicto"]


def detectar_idioma(texto: str) -> str:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(texto[:600])
    except Exception:
        return "?"


def cargar_traducciones(ruta: pathlib.Path) -> dict[str, dict[str, str]]:
    """Translations keyed by study, or empty when none were produced yet."""
    if not ruta.is_file():
        return {}
    with ruta.open(encoding="utf-8", newline="") as handle:
        return {fila["StudyInstanceUID"]: fila for fila in csv.DictReader(handle)}


def texto_a_extraer(estudio: Study, idioma: str, traducciones: dict) -> tuple[str, bool]:
    """Pick the report the extractor should read, and say whether it was translated."""
    traduccion = traducciones.get(estudio.uid)
    if idioma in IDIOMAS_NATIVOS or traduccion is None:
        return estudio.report, False
    return traduccion["reporte_en"], True


def fila_de(estudio: Study, extractor: ExtractorReglas,
            traducciones: dict) -> dict[str, object]:
    idioma = detectar_idioma(estudio.report)
    texto, traducido = texto_a_extraer(estudio, idioma, traducciones)
    extraccion = extractor.extraer(texto)
    fila: dict[str, object] = {
        "StudyInstanceUID": estudio.uid,
        "idioma": idioma,
        "traducido": int(traducido),
        "n_oraciones": len(dividir_en_oraciones(texto)),
        "n_evidencias": len(extraccion.evidencias),
        "es_dorado": int(estudio.has_gold_labels),
        "veredicto": ("con_hallazgo" if extraccion.positivos()
                      else clasificar(extraccion, texto).value),
    }
    fila.update(extraccion.etiquetas)
    return fila


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=pathlib.Path, default=TRAIN_CSV)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--traducidos", type=pathlib.Path, default=TRADUCIDOS)
    args = parser.parse_args(argv)

    estudios = load_studies(args.train_csv)
    extractor = ExtractorReglas()
    traducciones = cargar_traducciones(args.traducidos)
    if not traducciones:
        print("AVISO: no hay traducciones; se extrae del texto original en "
              "todos los idiomas. Corre traducir_reportes.py primero.")


    comenzado = time.time()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    # Archivo temporal + rename: una corrida interrumpida no deja un CSV a
    # medias que el pipeline tome por completo.
    parcial = args.salida.with_suffix(args.salida.suffix + ".parcial")
    with parcial.open("w", encoding="utf-8", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=CAMPOS_META + list(LABELS))
        escritor.writeheader()
        for indice, estudio in enumerate(estudios, start=1):
            escritor.writerow(fila_de(estudio, extractor, traducciones))
            if indice % 1000 == 0:
                print(f"  {indice:>5,}/{len(estudios):,}", flush=True)

    parcial.replace(args.salida)
    print(f"\n{len(estudios):,} estudios -> {args.salida} "
          f"({time.time() - comenzado:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
