"""Comprueba que dos corridas idénticas dan exactamente el mismo resultado.

Es la prueba que faltaba. Este proyecto reportó durante un tiempo una ablación
como significativa a partir de dos corridas que no eran reproducibles: la misma
configuración, con la misma semilla y los mismos datos, dio AUC macro de 0.743
y de 0.707 en dos ejecuciones distintas. Sembrar los generadores no alcanza en
GPU, porque cuDNN elige kernels en tiempo de ejecución y varios acumulan en
orden no determinista.

Este script entrena dos veces la misma configuración, en el mismo proceso, y
compara época por época. Si algo difiere, lo dice con el número exacto en vez
de con un "parece igual".

Usage:
    python pregunta1/paso3_modelos/verificar_determinismo.py
    python pregunta1/paso3_modelos/verificar_determinismo.py --epocas 5
    python pregunta1/paso3_modelos/verificar_determinismo.py --sin-determinismo
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pregunta1.paso3_modelos.entrenamiento import (  # noqa: E402
    ConfigEntrenamiento, Entrenador, fijar_semilla,
)

SALIDA = _RAIZ / "pregunta1/paso3_modelos/determinismo.json"


def _una_corrida(config: ConfigEntrenamiento, determinista: bool,
                 modelo_nombre: str, cache: pathlib.Path,
                 seleccion: pathlib.Path) -> list[dict]:
    """Entrena una vez y devuelve el resumen de cada época.

    Reproduce exactamente la secuencia de ``entrenar.py``: la semilla primero,
    y recién después la partición, los datasets y el modelo. El orden importa
    —el barajado y los pesos iniciales consumen el generador en ese orden—, y
    cambiarlo acá haría que la verificación no dijera nada sobre la corrida
    real.
    """
    from pregunta1.paso3_modelos.datos import DatasetRodilla, cargar_registros
    from pregunta1.paso3_modelos.entrenar import TRAIN_CSV, etiquetas_doradas
    from pregunta1.paso3_modelos.modelos import construir_modelo
    from pregunta1.paso3_modelos.particion import particionar
    from pregunta1.paso3_modelos.perdidas import construir_perdida

    fijar_semilla(config.semilla, determinista=determinista)

    registros = cargar_registros(seleccion, cache, etiquetas_doradas(TRAIN_CSV))
    particion = particionar(registros, semilla=config.semilla)
    entrenamiento = DatasetRodilla(particion.entrenamiento,
                                   aumentar=config.aumentar,
                                   semilla=config.semilla)
    validacion = DatasetRodilla(particion.validacion)

    modelo = construir_modelo(modelo_nombre, pooling=config.pooling)
    criterio = construir_perdida(
        config.perdida, list(entrenamiento.prevalencia.values()))

    entrenador = Entrenador(modelo, criterio, config)
    historial = entrenador.ajustar(
        DataLoader(entrenamiento, batch_size=config.lote, shuffle=True),
        DataLoader(validacion, batch_size=config.lote),
    )
    return [
        {"epoca": e.epoca,
         "perdida_entrenamiento": e.perdida_entrenamiento,
         "perdida_validacion": e.perdida_validacion,
         "auc_macro": e.auc_validacion.macro}
        for e in historial.epocas
    ]


def comparar(a: Sequence[dict], b: Sequence[dict]) -> dict:
    """Primera época en que las dos corridas difieren, y en cuánto."""
    if len(a) != len(b):
        return {"identicas": False, "motivo": "distinta cantidad de épocas",
                "epocas_a": len(a), "epocas_b": len(b)}

    campos = ("perdida_entrenamiento", "perdida_validacion", "auc_macro")
    peor = {"campo": None, "diferencia": 0.0, "epoca": None}
    primera = None
    for ea, eb in zip(a, b):
        for campo in campos:
            diferencia = abs(ea[campo] - eb[campo])
            if diferencia > 0 and primera is None:
                primera = ea["epoca"]
            if diferencia > peor["diferencia"]:
                peor = {"campo": campo, "diferencia": diferencia,
                        "epoca": ea["epoca"]}

    return {
        "identicas": primera is None,
        "primera_epoca_que_difiere": primera,
        "mayor_diferencia": peor,
        "auc_final_a": a[-1]["auc_macro"],
        "auc_final_b": b[-1]["auc_macro"],
        "mejor_auc_a": max(e["auc_macro"] for e in a),
        "mejor_auc_b": max(e["auc_macro"] for e in b),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=4,
                        help="pocas alcanzan: si divergen, lo hacen enseguida")
    parser.add_argument("--modelo", default="resnet50")
    parser.add_argument("--pooling", default="atencion")
    parser.add_argument("--cache", type=pathlib.Path,
                        default=_RAIZ / "pregunta1/paso2_preprocesamiento/cache")
    parser.add_argument("--seleccion", type=pathlib.Path,
                        default=_RAIZ / "pregunta1/paso2_preprocesamiento/estudios_seleccionados.csv")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--sin-determinismo", action="store_true",
                        help="mostrar que SIN el arreglo las corridas difieren")
    args = parser.parse_args(argv)

    determinista = not args.sin_determinismo
    config = ConfigEntrenamiento(epocas=args.epocas, pooling=args.pooling,
                                 paciencia=args.epocas)

    print(f"dispositivo: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'} | "
          f"determinista={determinista} | {args.epocas} épocas\n")

    print("corrida 1...", flush=True)
    primera = _una_corrida(config, determinista, args.modelo, args.cache,
                           args.seleccion)
    print("corrida 2...", flush=True)
    segunda = _una_corrida(config, determinista, args.modelo, args.cache,
                           args.seleccion)

    informe = comparar(primera, segunda) | {
        "determinista": determinista,
        "epocas": args.epocas,
        "pooling": args.pooling,
        "semilla": config.semilla,
    }

    print(f"\n{'época':>6} {'AUC corrida 1':>14} {'AUC corrida 2':>14} "
          f"{'diferencia':>12}")
    for ea, eb in zip(primera, segunda):
        print(f"{ea['epoca']:>6} {ea['auc_macro']:>14.10f} "
              f"{eb['auc_macro']:>14.10f} "
              f"{abs(ea['auc_macro'] - eb['auc_macro']):>12.2e}")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    if informe["identicas"]:
        print("\nIDÉNTICAS: las dos corridas coinciden bit a bit.")
        return 0

    peor = informe["mayor_diferencia"]
    print(f"\nDIFIEREN desde la época {informe['primera_epoca_que_difiere']}. "
          f"Mayor diferencia: {peor['diferencia']:.2e} en {peor['campo']} "
          f"(época {peor['epoca']}).")
    print(f"Mejor AUC: {informe['mejor_auc_a']:.4f} contra "
          f"{informe['mejor_auc_b']:.4f}.")
    # Sin determinismo esto es el resultado esperado, no una falla del script.
    return 0 if args.sin_determinismo else 1


if __name__ == "__main__":
    raise SystemExit(main())
