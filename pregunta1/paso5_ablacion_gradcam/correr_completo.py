"""Entrena las dos arquitecturas sobre el conjunto completo, por configuración.

Es el experimento que junta las dos palancas que quedaban sin medir:

* **Más estudios** — de 413 a 2 688 de entrenamiento, 6.5 veces más. Es lo que
  el enunciado dejaba abierto al sugerir "entre 300 y 600 estudios": la muestra
  chica era una decisión, no una restricción.
* **Más cortes por estudio** — 24, 72 y 192, contra los 24 de hoy. La cobertura
  medida dice que 24 ya cubre la banda central entera en el 93% de las series,
  así que la expectativa es que no cambie nada; medirlo es la única forma de
  saberlo.

Corre en orden creciente de costo. Si la configuración más cara falla —y es la
candidata, porque con 64 grupos el modelo procesa 512 imágenes por lote— las
dos baratas ya quedaron medidas.

Las corridas llevan prefijo ``completo_`` y caché propio, de modo que los
resultados ya publicados sobre la muestra de 573 estudios siguen
reproduciéndose sin tocar nada.

Usage:
    python pregunta1/paso5_ablacion_gradcam/correr_completo.py
    python pregunta1/paso5_ablacion_gradcam/correr_completo.py --grupos 8 24
    python pregunta1/paso5_ablacion_gradcam/correr_completo.py --solo-evaluar
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

PYTHON = sys.executable
SELECCION = _RAIZ / "pregunta1/paso2_preprocesamiento/estudios_completo.csv"
CACHES = _RAIZ / "pregunta1/paso2_preprocesamiento/caches_ablacion"
CORRIDAS = _RAIZ / "pregunta1/paso3_modelos/corridas"
SALIDA = _RAIZ / "pregunta1/paso5_ablacion_gradcam/completo.json"

GRUPOS = (8, 24, 64)
ARQUITECTURAS = ("resnet50", "inception")
POOLING = "atencion"      # con la que están definidas en el entregable #2
POOLINGS = ("atencion", "promedio")


def nombre_cache(grupos: int) -> pathlib.Path:
    return CACHES / f"completo_g{grupos:02d}_c{grupos * 3}"


def nombre_corrida(modelo: str, grupos: int, pooling: str = POOLING) -> str:
    sufijo = "" if pooling == "atencion" else f"_{pooling}"
    return f"completo_{modelo}_g{grupos:02d}_c{grupos * 3}{sufijo}"


def correr(orden: Sequence[str], titulo: str) -> tuple[int, float]:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}", flush=True)
    comenzado = time.time()
    codigo = subprocess.run(list(orden), cwd=_RAIZ).returncode
    return codigo, (time.time() - comenzado) / 60


def construir_cache(grupos: int) -> Optional[float]:
    destino = _RAIZ / nombre_cache(grupos)
    esperados = sum(1 for _ in (_RAIZ / SELECCION).open(encoding="utf-8")) - 1
    if destino.is_dir() and len(list(destino.glob("*.npy"))) >= esperados:
        print(f"\ncaché g={grupos}: completo, se salta")
        return 0.0
    codigo, minutos = correr(
        [PYTHON, "pregunta1/paso2_preprocesamiento/generar_cache.py",
         "--grupos", str(grupos), "--cache", str(nombre_cache(grupos)),
         "--seleccion", str(SELECCION)],
        f"caché g={grupos} ({grupos * 3} cortes)")
    return None if codigo else minutos


def entrenar(modelo: str, grupos: int, epocas: int,
             pooling: str = POOLING) -> Optional[float]:
    nombre = nombre_corrida(modelo, grupos, pooling)
    if (_RAIZ / CORRIDAS / nombre / "historial.json").is_file():
        print(f"\n{nombre}: ya entrenado, se salta")
        return 0.0
    codigo, minutos = correr(
        [PYTHON, "pregunta1/paso3_modelos/entrenar.py",
         "--modelo", modelo, "--pooling", pooling,
         "--epocas", str(epocas),
         "--cache", str(nombre_cache(grupos)),
         "--seleccion", str(SELECCION),
         "--etiqueta", nombre],
        f"entrenar {nombre}  ({grupos * 3} cortes, 2 688 estudios)")
    return None if codigo else minutos


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grupos", type=int, nargs="*", default=list(GRUPOS))
    parser.add_argument("--modelo", nargs="*", choices=list(ARQUITECTURAS),
                        default=list(ARQUITECTURAS))
    parser.add_argument("--pooling", nargs="*", choices=list(POOLINGS),
                        default=[POOLING])
    parser.add_argument("--epocas", type=int, default=25)
    parser.add_argument("--solo-cache", action="store_true")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    informe = {}
    if args.salida.is_file():
        informe = json.loads(args.salida.read_text(encoding="utf-8"))

    print(f"CONJUNTO COMPLETO · {len(args.grupos)} configuraciones × "
          f"{len(args.modelo)} arquitecturas")
    print(f"2 688 entrenamiento · 671 validación · 56 prueba (dorados)\n")

    for grupos in sorted(args.grupos):
        minutos = construir_cache(grupos)
        if minutos is None:
            print(f"FALLÓ el caché g={grupos}")
            return 1

        if args.solo_cache:
            continue

        for modelo in args.modelo:
          for pooling in args.pooling:
            minutos = entrenar(modelo, grupos, args.epocas, pooling)
            if minutos is None:
                # Una configuración que no entra en memoria no debe tumbar las
                # demás: se anota y se sigue con la siguiente.
                print(f"FALLÓ {nombre_corrida(modelo, grupos, pooling)}")
                informe[nombre_corrida(modelo, grupos, pooling)] = {
                    "grupos": grupos, "modelo": modelo, "fallo": True}
                continue
            informe[nombre_corrida(modelo, grupos, pooling)] = {
                "grupos": grupos, "cortes": grupos * 3, "modelo": modelo,
                "pooling": pooling, "minutos": round(minutos, 1)}
            args.salida.parent.mkdir(parents=True, exist_ok=True)
            args.salida.write_text(
                json.dumps(informe, indent=2, ensure_ascii=False),
                encoding="utf-8")

    print(f"\nescrito en {args.salida}")
    print("evaluá con: pregunta1/paso4_evaluacion/evaluar.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
