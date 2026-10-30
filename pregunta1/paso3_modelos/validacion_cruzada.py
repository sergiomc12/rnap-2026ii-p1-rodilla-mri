"""Validación cruzada de 5 pliegues sobre las etiquetas débiles.

Hasta ahora las decisiones se tomaron sobre una única partición 80/20 de
520 estudios con etiqueta débil. Con ese tamaño, el AUC de validación de un
pliegue cualquiera arrastra una varianza considerable, y elegir entre dos
configuraciones que difieren en pocos puntos equivale a leer ruido.

Este script corre el mismo modelo sobre los cinco pliegues y reporta la
media y la desviación. Los estudios dorados quedan fuera de todos los
pliegues, como en el resto del proyecto.

Usage:
    python pregunta1/paso3_modelos/validacion_cruzada.py --modelo resnet50
    python pregunta1/paso3_modelos/validacion_cruzada.py --modelo inception --epocas 15
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from torch.utils.data import DataLoader  # noqa: E402

from pregunta1.paso3_modelos.datos import DatasetRodilla, cargar_registros  # noqa: E402
from pregunta1.paso3_modelos.entrenamiento import (  # noqa: E402
    ConfigEntrenamiento, Entrenador, fijar_semilla,
)
from pregunta1.paso3_modelos.entrenar import (  # noqa: E402
    CACHE, SELECCION, TRAIN_CSV, etiquetas_doradas,
)
from pregunta1.paso3_modelos.modelos import construir_modelo  # noqa: E402
from pregunta1.paso3_modelos.particion import pliegues  # noqa: E402
from pregunta1.paso3_modelos.perdidas import construir_perdida  # noqa: E402

SALIDA = _RAIZ / "pregunta1/paso3_modelos/validacion_cruzada.json"


def correr_pliegue(particion, args, config: ConfigEntrenamiento) -> dict:
    """Entrena sobre un pliegue y devuelve su mejor AUC de validación."""
    fijar_semilla(config.semilla)

    entrenamiento = DatasetRodilla(particion.entrenamiento, aumentar=config.aumentar,
                                   semilla=config.semilla)
    validacion = DatasetRodilla(particion.validacion)

    modelo = construir_modelo(args.modelo, pooling=config.pooling,
                              preentrenado=not args.sin_preentrenar)
    criterio = construir_perdida(config.perdida,
                                 list(entrenamiento.prevalencia.values()))

    entrenador = Entrenador(modelo, criterio, config)
    comenzado = time.time()
    historial = entrenador.ajustar(
        DataLoader(entrenamiento, batch_size=config.lote, shuffle=True),
        DataLoader(validacion, batch_size=config.lote),
    )
    return {
        "auc": historial.mejor_auc,
        "mejor_epoca": historial.mejor_epoca,
        "epocas_corridas": len(historial.epocas),
        "minutos": round((time.time() - comenzado) / 60, 1),
        "n_entrenamiento": len(particion.entrenamiento),
        "n_validacion": len(particion.validacion),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", choices=["inception", "resnet50"], required=True)
    parser.add_argument("--perdida", default="bce",
                        choices=["bce", "bce_ponderada", "focal"])
    parser.add_argument("--pooling", default="atencion",
                        choices=["atencion", "atencion_sin_compuerta", "promedio"])
    parser.add_argument("--epocas", type=int, default=25)
    parser.add_argument("--pliegues", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-preentrenar", action="store_true")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--cache", type=pathlib.Path, default=CACHE,
                        help="cambialo para ablacionar el preprocesamiento")
    parser.add_argument("--etiqueta", help="nombre con el que guardar")
    parser.add_argument("--rehacer", action="store_true",
                        help="recalcular aunque esta configuracion ya este")
    args = parser.parse_args(argv)

    config = ConfigEntrenamiento(
        epocas=args.epocas, perdida=args.perdida, pooling=args.pooling,
        semilla=args.semilla,
    )

    nombre = args.etiqueta or f"{args.modelo}_{config.perdida}_{config.pooling}"

    # El JSON acumula varias configuraciones, de modo que su existencia no dice
    # si ESTA ya se calculo. Sin este chequeo, el comando unico reentrenaba
    # cinco pliegues en cada corrida para cada configuracion sin artefacto
    # propio: veinte minutos para reescribir lo que ya estaba.
    if not args.rehacer and args.salida.is_file():
        previos = json.loads(args.salida.read_text(encoding="utf-8"))
        if nombre in previos:
            anterior = previos[nombre]
            print(f"{nombre} ya esta en {args.salida}: "
                  f"AUC {anterior['auc_media']:.4f} +/- "
                  f"{anterior['auc_desviacion']:.4f} sobre "
                  f"{anterior['pliegues']} pliegues. Usa --rehacer para "
                  f"recalcularlo.")
            return 0

    registros = cargar_registros(SELECCION, args.cache,
                                 etiquetas_doradas(TRAIN_CSV))
    particiones = pliegues(registros, n_pliegues=args.pliegues, semilla=args.semilla)

    print(f"VALIDACION CRUZADA · {args.pliegues} pliegues · {nombre}")
    print(f"{len(registros)} estudios | "
          f"{len(particiones[0].prueba)} dorados fuera de todos los pliegues\n")

    resultados = []
    comenzado = time.time()
    for indice, particion in enumerate(particiones, start=1):
        resultado = correr_pliegue(particion, args, config)
        resultados.append(resultado)
        print(f"  pliegue {indice}/{args.pliegues} | "
              f"train={resultado['n_entrenamiento']} val={resultado['n_validacion']} | "
              f"AUC {resultado['auc']:.4f} | epoca {resultado['mejor_epoca']} | "
              f"{resultado['minutos']:.1f} min", flush=True)

    aucs = [r["auc"] for r in resultados]
    media = statistics.mean(aucs)
    desviacion = statistics.stdev(aucs) if len(aucs) > 1 else 0.0

    print(f"\n  AUC validacion: {media:.4f} +/- {desviacion:.4f}")
    print(f"  rango         : [{min(aucs):.4f}, {max(aucs):.4f}]")
    print(f"  total         : {(time.time()-comenzado)/60:.1f} min")

    _guardar(args, config, nombre, resultados, media, desviacion)
    return 0


def _guardar(args, config, nombre, resultados, media, desviacion) -> None:
    """Acumula en el JSON en vez de pisarlo, para comparar configuraciones."""
    previos = {}
    if args.salida.is_file():
        previos = json.loads(args.salida.read_text(encoding="utf-8"))

    previos[nombre] = {
        "modelo": args.modelo,
        "perdida": config.perdida,
        "pooling": config.pooling,
        "preentrenado": not args.sin_preentrenar,
        "pliegues": args.pliegues,
        "auc_media": media,
        "auc_desviacion": desviacion,
        "auc_min": min(r["auc"] for r in resultados),
        "auc_max": max(r["auc"] for r in resultados),
        "por_pliegue": resultados,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(previos, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"  guardado en {args.salida}")


if __name__ == "__main__":
    raise SystemExit(main())
