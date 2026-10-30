"""Ablación del número de cortes que el modelo ve por estudio.

El preprocesamiento toma ``grupos`` ventanas de ``cortes_por_grupo`` cortes
contiguos, repartidas sobre la banda central de la serie sagital. Con la
configuración por defecto —8 grupos de 3— el modelo ve hasta 24 cortes
distintos, lo que cubre la banda entera en el 91% de las series pero deja fuera
parte de las largas.

Si eso importa o no es una pregunta empírica, y se venía respondiendo por
razonamiento. Este script la mide sobre **las dos arquitecturas del entregable
\\#2**: construye un caché por configuración de cortes —compartido entre
arquitecturas, porque el preprocesamiento no depende del modelo—, entrena cada
arquitectura sobre cada caché y corre validación cruzada de cinco pliegues.

La comparación se decide en **validación cruzada**, no sobre los dorados. Con
56 estudios el intervalo es de ±0.05 y ninguna diferencia plausible entre
configuraciones de preprocesamiento sería visible; con cinco pliegues sobre 520
estudios la desviación ronda 0.014, que es la resolución que hace falta.

Cada arquitectura conserva la agregación con la que está definida en el
entregable (atención), de modo que dentro de un brazo lo único que cambia es
cuántos cortes ve el modelo.

Es reanudable: una configuración ya entrenada se salta, así que relanzarlo tras
un corte sigue donde quedó.

Usage:
    python pregunta1/paso5_ablacion_gradcam/ablacion_cortes.py
    python pregunta1/paso5_ablacion_gradcam/ablacion_cortes.py --solo-cache
    python pregunta1/paso5_ablacion_gradcam/ablacion_cortes.py --grupos 8 13
    python pregunta1/paso5_ablacion_gradcam/ablacion_cortes.py --modelo resnet50
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

PYTHON = sys.executable
CACHES = _RAIZ / "pregunta1/paso2_preprocesamiento/caches_ablacion"
CORRIDAS = _RAIZ / "pregunta1/paso3_modelos/corridas"
CRUZADA = _RAIZ / "pregunta1/paso3_modelos/validacion_cruzada.json"
SALIDA = _RAIZ / "pregunta1/paso5_ablacion_gradcam/ablacion_cortes.json"
CORTES_POR_GRUPO = 3
PREFIJO = "cortes"
SELECCION = None

# Las dos arquitecturas del entregable #2, cada una con su agregación.
ARQUITECTURAS = {
    "inception": "atencion",
    "resnet50": "atencion",
}


@dataclass(frozen=True)
class Config:
    """Una configuración de cortes."""

    grupos: int
    nota: str

    @property
    def cortes(self) -> int:
        """Cortes distintos como mucho: g ventanas de 3 contiguos."""
        return self.grupos * CORTES_POR_GRUPO

    @property
    def etiqueta(self) -> str:
        return f"g{self.grupos:02d}_c{self.cortes}"

    @property
    def cache(self) -> pathlib.Path:
        # Con la muestra por defecto, g=8 reusa el caché que ya existe en vez
        # de duplicar 2.8 GB para reprocesar exactamente lo mismo.
        if self.grupos == 8 and SELECCION is None:
            return _RAIZ / "pregunta1/paso2_preprocesamiento/cache"
        return CACHES / f"{PREFIJO}_{self.etiqueta}"

    def corrida(self, modelo: str) -> str:
        return f"{PREFIJO}_{modelo}_{self.etiqueta}"


CONFIGS = (
    Config(8, "actual: cubre la banda entera en el 91% de las series"),
    Config(13, "~40 cortes: el umbral donde g=8 empieza a quedarse corto"),
    Config(17, "~50 cortes"),
    Config(20, "~60 cortes"),
    Config(24, "72 cortes"),
    Config(64, "cubre la banda de la serie más larga del conjunto (320 cortes)"),
)


def correr(orden: Sequence[str], titulo: str) -> tuple[int, float]:
    print(f"\n--- {titulo}", flush=True)
    comenzado = time.time()
    codigo = subprocess.run(list(orden), cwd=_RAIZ).returncode
    return codigo, (time.time() - comenzado) / 60


def construir_cache(config: Config, rehacer: bool) -> Optional[float]:
    """Genera el caché de una configuración. Devuelve los minutos que costó."""
    # El atajo vale solo para la muestra por defecto: ahí g=8 es exactamente
    # el caché que el paso 2 ya construyó. Con otra selección hay que
    # generarlo, y saltarlo dejaría entrenando sobre los estudios equivocados.
    if config.grupos == 8 and SELECCION is None:
        print(f"\n--- caché g=8: se reusa {config.cache}")
        return 0.0

    destino = _RAIZ / config.cache
    if destino.is_dir() and any(destino.iterdir()) and not rehacer:
        print(f"\n--- caché {config.etiqueta}: ya existe, se salta")
        return 0.0

    codigo, minutos = correr(
        [PYTHON, "pregunta1/paso2_preprocesamiento/generar_cache.py",
         "--grupos", str(config.grupos), "--cache", str(config.cache),
         *(("--seleccion", str(SELECCION)) if SELECCION else ())],
        f"caché {config.etiqueta} ({config.cortes} cortes)")
    return None if codigo else minutos


def entrenar(config: Config, modelo: str, pooling: str,
             epocas: int) -> Optional[float]:
    """Una corrida de esta arquitectura sobre el caché de esta configuración."""
    nombre = config.corrida(modelo)
    if (_RAIZ / CORRIDAS / nombre / "historial.json").is_file():
        print(f"\n--- entrenar {nombre}: ya existe, se salta")
        return 0.0
    codigo, minutos = correr(
        [PYTHON, "pregunta1/paso3_modelos/entrenar.py",
         "--modelo", modelo, "--pooling", pooling,
         "--epocas", str(epocas), "--cache", str(config.cache),
         *(("--seleccion", str(SELECCION)) if SELECCION else ()),
         "--etiqueta", nombre],
        f"entrenar {nombre}")
    return None if codigo else minutos


def validar(config: Config, modelo: str, pooling: str, epocas: int,
            pliegues: int) -> Optional[float]:
    """Validación cruzada: la medición que decide la comparación."""
    codigo, minutos = correr(
        [PYTHON, "pregunta1/paso3_modelos/validacion_cruzada.py",
         "--modelo", modelo, "--pooling", pooling,
         "--epocas", str(epocas), "--pliegues", str(pliegues),
         "--cache", str(config.cache),
         "--etiqueta", config.corrida(modelo)],
        f"validación cruzada {config.corrida(modelo)}")
    return None if codigo else minutos


def _tamano_gb(ruta: pathlib.Path) -> float:
    destino = _RAIZ / ruta
    if not destino.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in destino.glob("*.npy")) / 1e9


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grupos", type=int, nargs="*",
                        help="solo estas configuraciones de cortes")
    parser.add_argument("--modelo", nargs="*", choices=list(ARQUITECTURAS),
                        default=list(ARQUITECTURAS))
    parser.add_argument("--epocas", type=int, default=25)
    parser.add_argument("--pliegues", type=int, default=5)
    parser.add_argument("--solo-cache", action="store_true")
    parser.add_argument("--sin-cruzada", action="store_true",
                        help="solo una corrida por configuración, sin pliegues")
    parser.add_argument("--rehacer-cache", action="store_true")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    # Con --seleccion se corre sobre otro conjunto de estudios; el prefijo
    # mantiene separados sus caches y sus corridas de los de la muestra
    # chica, para no pisar los resultados ya reportados.
    parser.add_argument("--seleccion", type=pathlib.Path)
    parser.add_argument("--prefijo", default="cortes")
    args = parser.parse_args(argv)
    global PREFIJO, SELECCION
    PREFIJO = args.prefijo
    SELECCION = args.seleccion

    configs = [c for c in CONFIGS
               if not args.grupos or c.grupos in args.grupos]
    print(f"ABLACIÓN DE CORTES · {len(configs)} configuraciones × "
          f"{len(args.modelo)} arquitecturas\n")
    for c in configs:
        print(f"  g={c.grupos:>2}  hasta {c.cortes:>3} cortes  {c.nota}")
    print()
    for m in args.modelo:
        print(f"  {m} + {ARQUITECTURAS[m]}")

    informe = {}
    if args.salida.is_file():
        informe = json.loads(args.salida.read_text(encoding="utf-8"))

    for config in configs:
        minutos = construir_cache(config, args.rehacer_cache)
        if minutos is None:
            print(f"FALLÓ el caché de {config.etiqueta}")
            return 1
        comun = {"grupos": config.grupos, "cortes": config.cortes,
                 "nota": config.nota, "cache": str(config.cache),
                 "minutos_cache": round(minutos, 1),
                 "cache_gb": round(_tamano_gb(config.cache), 2)}

        if args.solo_cache:
            informe[config.etiqueta] = comun
            _guardar(args.salida, informe)
            continue

        for modelo in args.modelo:
            pooling = ARQUITECTURAS[modelo]
            fila = dict(comun, modelo=modelo, pooling=pooling)

            minutos = entrenar(config, modelo, pooling, args.epocas)
            if minutos is None:
                print(f"FALLÓ el entrenamiento de {config.corrida(modelo)}")
                return 1
            fila["minutos_entrenamiento"] = round(minutos, 1)

            if not args.sin_cruzada:
                minutos = validar(config, modelo, pooling, args.epocas,
                                  args.pliegues)
                if minutos is None:
                    print(f"FALLÓ la validación cruzada de "
                          f"{config.corrida(modelo)}")
                    return 1
                fila["minutos_cruzada"] = round(minutos, 1)

            informe[config.corrida(modelo)] = fila
            _guardar(args.salida, informe)

    print(f"\nescrito en {args.salida}")
    return 0


def _guardar(ruta: pathlib.Path, informe: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(informe, indent=2, ensure_ascii=False),
                    encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
