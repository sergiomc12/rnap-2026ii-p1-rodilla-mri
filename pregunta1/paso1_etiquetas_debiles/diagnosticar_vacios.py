"""Por qué el extractor deja 807 estudios sin ninguna etiqueta.

Un estudio vacío no es una rodilla sana: de los 58 estudios con etiqueta de
radiólogo, **ninguno** tiene las doce en cero —todos traen entre 1 y 9
hallazgos—. Son pacientes a los que se les hizo una resonancia. Así que un
estudio vacío es, casi siempre, una falla de extracción, y este script la
clasifica en vez de suponerla.

Separa tres causas, que piden arreglos distintos:

* **sin evidencia** — ninguna oración mencionó ningún hallazgo. El léxico no
  reconoce la terminología, o la traducción la destruyó.
* **todo negado** — se encontró evidencia y toda quedó clasificada como negada.
  Puede ser correcto (informe que descarta todo) o un disparador de negación
  que abarca de más.
* **todo incierto** — la evidencia quedó en "posible/probable", que por
  configuración no cuenta como positivo.

Usage:
    python pregunta1/paso1_etiquetas_debiles/diagnosticar_vacios.py
    python pregunta1/paso1_etiquetas_debiles/diagnosticar_vacios.py --ejemplos 5
    python pregunta1/paso1_etiquetas_debiles/diagnosticar_vacios.py --idioma el
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys
from typing import Optional, Sequence

# pregunta1/ entra al sys.path para poder importar esta pregunta como
# paquete: los scripts se corren por ruta, no con -m. Las rutas de datos
# salen de pregunta1/rutas.py: siempre dentro de pregunta1/datos/.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

from pregunta1 import rutas as _rutas  # noqa: E402

from pregunta1.paso1_etiquetas_debiles.extractor import ExtractorReglas  # noqa: E402
from pregunta1.paso1_etiquetas_debiles.negacion import EstadoAsercion  # noqa: E402

TRAIN_CSV = _rutas.TRAIN_CSV
TRADUCIDOS = _RAIZ / "pregunta1/paso1_etiquetas_debiles/reportes_traducidos.csv"
ETIQUETAS = _RAIZ / "pregunta1/paso1_etiquetas_debiles/etiquetas_debiles.csv"
SALIDA = _RAIZ / "pregunta1/paso1_etiquetas_debiles/diagnostico_vacios.json"


def cargar_textos() -> dict[str, tuple[str, str]]:
    """Texto que el extractor realmente leyó, por estudio: (texto, idioma).

    Para los idiomas que se traducen, lo que el extractor ve es la traducción,
    no el original. Diagnosticar sobre el original daría una causa equivocada.
    """
    idioma = {}
    for fila in csv.DictReader(ETIQUETAS.open(encoding="utf-8")):
        idioma[fila["StudyInstanceUID"]] = fila["idioma"]

    textos = {}
    for fila in csv.DictReader(TRAIN_CSV.open(encoding="utf-8", errors="replace")):
        textos[fila["StudyInstanceUID"]] = fila["Report"]

    # La estrategia elegida en el paso 1 es hibrida: se traduce TODO salvo
    # ingles y espanol, que el extractor lee de forma nativa. Diagnosticar
    # sobre la traduccion del espanol daria una causa que no existe.
    if TRADUCIDOS.is_file():
        for fila in csv.DictReader(TRADUCIDOS.open(encoding="utf-8")):
            uid = fila["StudyInstanceUID"]
            if idioma.get(uid) in ("en", "es"):
                continue
            traducido = fila.get("reporte_en", "")
            if traducido.strip():
                textos[uid] = traducido

    return {uid: (texto, idioma.get(uid, "?")) for uid, texto in textos.items()}


def clasificar(extraccion) -> str:
    """Por qué este estudio quedó sin ninguna etiqueta."""
    if not extraccion.evidencias:
        return "sin evidencia"
    estados = {e.estado for e in extraccion.evidencias}
    if estados == {EstadoAsercion.NEGADO}:
        return "todo negado"
    if estados == {EstadoAsercion.INCIERTO}:
        return "todo incierto"
    return "mezcla negado/incierto"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ejemplos", type=int, default=3,
                        help="oraciones de muestra por causa")
    parser.add_argument("--idioma", help="diagnosticar solo este idioma")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    textos = cargar_textos()
    extractor = ExtractorReglas()

    causas = collections.Counter()
    por_idioma: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    ejemplos: dict[str, list] = collections.defaultdict(list)
    terminos_bloqueados = collections.Counter()
    total = 0

    for uid, (texto, idioma) in textos.items():
        if args.idioma and idioma != args.idioma:
            continue
        extraccion = extractor.extraer(texto)
        if extraccion.positivos():
            continue

        total += 1
        causa = clasificar(extraccion)
        causas[causa] += 1
        por_idioma[idioma][causa] += 1

        for evidencia in extraccion.evidencias:
            terminos_bloqueados[(evidencia.etiqueta, evidencia.estado.value)] += 1

        if len(ejemplos[causa]) < args.ejemplos:
            ejemplos[causa].append({
                "estudio": uid, "idioma": idioma,
                "oraciones": len(texto.split(".")),
                "evidencias": [
                    {"etiqueta": e.etiqueta, "estado": e.estado.value,
                     "coincide": e.texto_coincidente,
                     "oracion": e.oracion[:180]}
                    for e in extraccion.evidencias[:4]
                ],
                "muestra_texto": texto[:220].replace("\n", " "),
            })

    print(f"ESTUDIOS SIN NINGUNA ETIQUETA: {total:,}\n")
    print(f"{'causa':24} {'n':>6} {'%':>7}")
    for causa, n in causas.most_common():
        print(f"{causa:24} {n:>6,} {n/total:>6.1%}")

    print(f"\n{'idioma':8} " + " ".join(f"{c[:12]:>14}" for c in causas))
    for idioma, cuenta in sorted(por_idioma.items(),
                                 key=lambda kv: -sum(kv[1].values())):
        fila = " ".join(f"{cuenta.get(c, 0):>14,}" for c in causas)
        print(f"{idioma:8} {fila}")

    if terminos_bloqueados:
        print("\nevidencia encontrada pero descartada (etiqueta, estado):")
        for (etiqueta, estado), n in terminos_bloqueados.most_common(10):
            print(f"  {etiqueta:18} {estado:10} {n:>5,}")

    for causa, casos in ejemplos.items():
        print(f"\n--- ejemplos de '{causa}'")
        for caso in casos:
            print(f"  [{caso['idioma']}] {caso['muestra_texto']}")
            for e in caso["evidencias"]:
                print(f"      {e['etiqueta']:18} {e['estado']:10} "
                      f"'{e['coincide']}'  <-  {e['oracion'][:110]}")

    args.salida.write_text(json.dumps({
        "total_sin_etiqueta": total,
        "causas": dict(causas),
        "por_idioma": {k: dict(v) for k, v in por_idioma.items()},
        "evidencia_descartada": {f"{a}|{b}": n
                                 for (a, b), n in terminos_bloqueados.items()},
        "ejemplos": {k: v for k, v in ejemplos.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
