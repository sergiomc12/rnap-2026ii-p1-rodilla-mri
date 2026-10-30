"""Train one architecture and save its weights and history.

Both architectures run through this same script, so the only thing that
differs between them is `--modelo`.

Usage:
    python pregunta1/paso3_modelos/entrenar.py --modelo inception
    python pregunta1/paso3_modelos/entrenar.py --modelo resnet50
    python pregunta1/paso3_modelos/entrenar.py --modelo resnet50 --perdida focal
"""

from __future__ import annotations

import argparse
import csv
import json
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

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pregunta1.paso3_modelos.datos import (  # noqa: E402
    LABELS, DatasetRodilla, cargar_registros,
)
from pregunta1.paso3_modelos.entrenamiento import (  # noqa: E402
    ConfigEntrenamiento, Entrenador, cargar_pesos, fijar_semilla,
)
from pregunta1.paso3_modelos.modelos import construir_modelo  # noqa: E402
from pregunta1.paso3_modelos.particion import particionar  # noqa: E402
from pregunta1.paso3_modelos.perdidas import construir_perdida  # noqa: E402

SELECCION = _RAIZ / "pregunta1/paso2_preprocesamiento/estudios_seleccionados.csv"
CACHE = _RAIZ / "pregunta1/paso2_preprocesamiento/cache"
TRAIN_CSV = _rutas.TRAIN_CSV
SALIDA = _RAIZ / "pregunta1/paso3_modelos/corridas"


def etiquetas_doradas(ruta: pathlib.Path) -> dict:
    """Radiologist labels for the gold studies, keyed by study UID.

    The selection CSV carries the extractor's guess for every study; for
    the 58 gold ones the real labels must win, since they are the test set.
    """
    with ruta.open(encoding="utf-8", newline="") as handle:
        reales = {}
        for fila in csv.DictReader(handle):
            valores = [fila[e].strip() for e in LABELS]
            if all(v != "" for v in valores):
                reales[fila["StudyInstanceUID"]] = [int(v) for v in valores]
    return reales


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True,
                        choices=["inception", "inception_residual", "inception_googlenet", "inception_googlenet_residual", "resnet50"])
    parser.add_argument("--perdida", default="bce",
                        choices=["bce", "bce_ponderada", "focal"])
    parser.add_argument("--pooling", default="atencion",
                        choices=["atencion", "atencion_sin_compuerta",
                                 "atencion_por_etiqueta",
                                 "atencion_por_etiqueta_sin_compuerta",
                                 "promedio"])
    parser.add_argument("--epocas", type=int, default=15)
    parser.add_argument("--lote", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--paciencia", type=int, default=5,
                        help="epocas sin mejorar antes de cortar. Subirlo importa cuando la corrida entrena POCAS etiquetas: el macro de dos es mas ruidoso que el de doce, y el corte se dispara por ruido. En la corrida larga de doce etiquetas hubo un tramo de 4 epocas sin mejorar seguido de una mejora")
    parser.add_argument("--sin-aumentar", action="store_true")
    parser.add_argument("--sin-preentrenar", action="store_true",
                        help="resnet50 desde cero, para la ablacion")
    parser.add_argument("--etiqueta", help="nombre de la corrida")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--seleccion", type=pathlib.Path, default=SELECCION,
                        help="CSV de estudios; cambialo para la ablacion de plano")
    parser.add_argument("--cache", type=pathlib.Path, default=CACHE)
    parser.add_argument("--acumular", type=int, default=1,
                        help="lotes a acumular antes de actualizar; el lote "
                             "EFECTIVO es lote*acumular, sin pagar su memoria")
    parser.add_argument("--multiplano", action="store_true",
                        help="apila sagital+coronal+axial en 192 grupos, para "
                             "que cada etiqueta elija su plano")
    parser.add_argument("--cache-coronal", type=pathlib.Path)
    parser.add_argument("--cache-axial", type=pathlib.Path)
    parser.add_argument("--amp", action="store_true",
                        help="precision mixta: las convoluciones corren en "
                             "float16 en los nucleos tensoriales. Los DATOS y "
                             "los PESOS siguen en float32; medido, la "
                             "diferencia en las 12 probabilidades es 0.000000")
    parser.add_argument("--reanudar", type=pathlib.Path,
                        help="pesos.pt de donde continuar. Carga los pesos (y "
                             "el estado del optimizador si el checkpoint lo "
                             "trae) en vez de arrancar de cero")
    parser.add_argument("--auc-previo", type=float, default=0.0,
                        help="el mejor AUC que ya tenia el checkpoint. Con "
                             "--reanudar se saca solo del progreso.json; sin "
                             "esto, la primera epoca pisa los pesos buenos "
                             "aunque sea peor")
    parser.add_argument("--epoca-previa", type=int, default=0,
                        help="en que epoca se logro --auc-previo. Sin esto se "
                             "asume la anterior al reinicio, lo que suele ser "
                             "falso: el mejor puede estar muchas epocas atras, "
                             "y el historial quedaria anunciando una epoca que "
                             "nunca fue la mejor")
    parser.add_argument("--epoca-inicial", type=int, default=1,
                        help="numero de la primera epoca; con --reanudar se "
                             "pone la siguiente a la ultima de la corrida "
                             "anterior, para que la curva se pueda pegar")
    parser.add_argument("--etiquetas", default="",
                        help="lista separada por comas de las etiquetas que "
                             "esta corrida entrena; el resto no aporta "
                             "gradiente. Vacio = las doce. Para el "
                             "entrenamiento por cara: cada plano recibe las "
                             "etiquetas que mejor le salen")
    parser.add_argument("--en-memoria", action="store_true",
                        help="sube todo el cache a RAM una vez; solo tiene "
                             "sentido con el cache en uint8 (39 GB) --en "
                             "float32 serian 156 GB y no entra")
    args = parser.parse_args(argv)

    # Se valida antes de tocar la GPU: un nombre mal escrito aca entrenaria
    # en silencio un subconjunto distinto del pedido, y el error recien se
    # veria al leer el AUC de una etiqueta que nunca se entreno.
    activas = tuple(e.strip() for e in args.etiquetas.split(",") if e.strip())
    desconocidas = [e for e in activas if e not in LABELS]
    if desconocidas:
        parser.error(f"etiquetas desconocidas: {desconocidas}; validas: {list(LABELS)}")
    indices_activos = [LABELS.index(e) for e in activas] if activas else None

    config = ConfigEntrenamiento(
        epocas=args.epocas, lote=args.lote, learning_rate=args.learning_rate,
        perdida=args.perdida, pooling=args.pooling,
        aumentar=not args.sin_aumentar, semilla=args.semilla,
        acumular=args.acumular, etiquetas_activas=activas,
        paciencia=args.paciencia, epoca_inicial=args.epoca_inicial,
        auc_previo=args.auc_previo,
        epoca_previa=(args.epoca_previa or max(0, args.epoca_inicial - 1)
                      if args.auc_previo else 0),
    )
    fijar_semilla(config.semilla)

    registros = cargar_registros(args.seleccion, args.cache,
                                 etiquetas_doradas(TRAIN_CSV))
    particion = particionar(registros, semilla=config.semilla)
    print(f"{len(registros)} estudios | {particion.resumen()}")

    if args.multiplano:
        # Los tres planos del mismo estudio apilados: (192, 3, 224, 224).
        # El cache sagital es el que se pasa en --cache; los otros dos se
        # derivan de --cache-coronal y --cache-axial.
        from pregunta1.paso3_modelos.datos_multiplano import DatasetMultiplano
        caches = {"sagital": args.cache, "coronal": args.cache_coronal,
                  "axial": args.cache_axial}
        entrenamiento = DatasetMultiplano(particion.entrenamiento, caches,
                                          aumentar=config.aumentar,
                                          semilla=config.semilla)
        validacion = DatasetMultiplano(particion.validacion, caches)
        print(f"multiplano: {len(entrenamiento)} train / {len(validacion)} val "
              f"con los tres planos")
    else:
        entrenamiento = DatasetRodilla(particion.entrenamiento, aumentar=config.aumentar,
                                       semilla=config.semilla,
                                       en_memoria=args.en_memoria)
        validacion = DatasetRodilla(particion.validacion, en_memoria=args.en_memoria)

    modelo = construir_modelo(args.modelo, pooling=config.pooling,
                              preentrenado=not args.sin_preentrenar)
    criterio = construir_perdida(
        config.perdida, list(entrenamiento.prevalencia.values()),
        indices_activos=indices_activos,
    )

    parametros = sum(p.numel() for p in modelo.parameters()) / 1e6
    print(f"modelo={args.modelo} perdida={config.perdida} pooling={config.pooling} "
          f"params={parametros:.1f}M")
    print(f"cache={args.cache}")
    print(f"seleccion={args.seleccion}")
    print(f"epocas={config.epocas} lote={config.lote} lr={config.learning_rate} "
          f"paciencia={config.paciencia} semilla={config.semilla} "
          f"aumentar={config.aumentar} amp={bool(args.amp)}")
    print(f"etiquetas_activas={list(config.etiquetas_activas) or 'las 12'}\n")

    entrenador = Entrenador(modelo, criterio, config, amp=args.amp)
    print(f"dispositivo: {entrenador.dispositivo}")

    if args.reanudar:
        # Despues de construir el Entrenador: el optimizador se crea ahi,
        # y restaurar sus momentos es justamente lo que evita el bajon de
        # las primeras epocas al continuar.
        info = cargar_pesos(args.reanudar, entrenador.modelo,
                            entrenador.optimizador)
        entrenador.modelo.to(entrenador.dispositivo)
        print(f"reanudado desde {args.reanudar}")
        if info["con_optimizador"]:
            print(f"  con estado del optimizador (venia de la epoca "
                  f"{info['epoca']})")
        else:
            print("  OJO: el checkpoint es del formato viejo y NO trae "
                  "el estado del optimizador.")
            print("  AdamW arranca con sus momentos en cero, asi que las "
                  "primeras epocas pueden bajar antes de recuperar.")
    print()

    def informar(resultado, es_mejor) -> None:
        marca = " *" if es_mejor else ""
        # Con un subconjunto se imprimen los dos: el de las activas es el que
        # decide la mejor epoca, el de las doce queda para no perder de vista
        # que las otras cabezas no se estan entrenando.
        auc = f"AUC {resultado.auc_validacion.macro:.4f}"
        if activas:
            auc = (f"AUC({len(activas)}) "
                   f"{resultado.auc_validacion.macro_de(activas):.4f} | "
                   f"AUC(12) {resultado.auc_validacion.macro:.4f}")
        ultima = config.epoca_inicial + config.epocas - 1
        print(f"  epoca {resultado.epoca:>2}/{ultima} | "
              f"train {resultado.perdida_entrenamiento:.4f} | "
              f"val {resultado.perdida_validacion:.4f} | "
              f"{auc} | "
              f"{resultado.segundos:.0f}s{marca}", flush=True)

    comenzado = time.time()
    # La carpeta de respaldo es la MISMA donde `_guardar` escribe al final, para
    # que un checkpoint intermedio y uno completo no queden en sitios distintos.
    destino = args.salida / (args.etiqueta
                             or f"{args.modelo}_{config.perdida}_{config.pooling}")
    _escribir_configuracion(destino, args, config)
    historial = entrenador.ajustar(
        DataLoader(entrenamiento, batch_size=config.lote, shuffle=True),
        DataLoader(validacion, batch_size=config.lote),
        al_terminar_epoca=informar,
        carpeta_respaldo=destino,
    )
    minutos = (time.time() - comenzado) / 60
    print(f"\nmejor epoca {historial.mejor_epoca} | "
          f"AUC validacion {historial.mejor_auc:.4f} | {minutos:.1f} min")

    _guardar(args, config, modelo, historial, parametros, minutos)
    return 0


def _escribir_configuracion(carpeta: pathlib.Path, args, config) -> None:
    """Deja la configuracion en disco ANTES de entrenar.

    `historial.json` --que es donde viven el cache y el CSV de seleccion-- se
    escribe recien al terminar, asi que una corrida interrumpida dejaba sus
    `pesos.pt` sin registro de con que datos se entrenaron. Eso ya costo caro
    una vez: un `historial.json` de otra corrida quedo en la carpeta de la
    corrida larga apuntando al cache uint8, asi que evaluar esos pesos
    float32 con la configuracion que decia el historial habria leido el cache
    equivocado --y el AUC resultante no habria significado nada--.

    Con esto, unos pesos sueltos siempre se pueden volver a evaluar con el
    mismo cache, la misma particion y las mismas etiquetas activas.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "configuracion.json").write_text(json.dumps({
        "modelo": args.modelo,
        "perdida": config.perdida,
        "pooling": config.pooling,
        "preentrenado": not args.sin_preentrenar,
        "seleccion": str(args.seleccion),
        "cache": str(args.cache),
        "cache_coronal": str(args.cache_coronal) if args.cache_coronal else None,
        "cache_axial": str(args.cache_axial) if args.cache_axial else None,
        "multiplano": args.multiplano,
        "etiquetas_activas": list(config.etiquetas_activas),
        "reanudado_de": str(args.reanudar) if args.reanudar else None,
        "epoca_inicial": config.epoca_inicial,
        "epocas": config.epocas,
        "lote": config.lote,
        "acumular": config.acumular,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "paciencia": config.paciencia,
        "aumentar": config.aumentar,
        "semilla": config.semilla,
        "amp": bool(args.amp),
        "en_memoria": bool(args.en_memoria),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def _guardar(args, config, modelo, historial, parametros, minutos) -> None:
    nombre = args.etiqueta or f"{args.modelo}_{config.perdida}_{config.pooling}"
    carpeta = args.salida / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    # Solo se escribe `pesos.pt` si ESTA corrida produjo la mejor epoca. Si no,
    # los pesos en memoria son los de la ultima epoca --`ajustar` no restaura
    # nada porque no hubo nada mejor que restaurar-- y guardarlos destruiria el
    # checkpoint de la fase anterior, que es el bueno.
    if historial.mejoro_aqui:
        torch.save(modelo.state_dict(), carpeta / "pesos.pt")
    else:
        print(f"pesos.pt NO se toca: la mejor epoca ({historial.mejor_epoca}, "
              f"AUC {historial.mejor_auc:.4f}) es de una corrida anterior y "
              f"ninguna de esta la supero.")
    (carpeta / "historial.json").write_text(json.dumps({
        "modelo": args.modelo,
        "perdida": config.perdida,
        "pooling": config.pooling,
        "preentrenado": not args.sin_preentrenar,
        "aumentar": config.aumentar,
        "semilla": config.semilla,
        # Con qué datos se entrenó. No es documentación: la ablación de plano
        # entrena sobre el caché coronal, y sin este dato la evaluación la
        # corría contra tensores sagitales —un modelo coronal mirando imágenes
        # de otro plano— y el número resultante no significaba nada.
        "seleccion": str(args.seleccion),
        "cache": str(args.cache),
        # Sin esto, un historial.json de una corrida por cara es
        # indistinguible de una de doce etiquetas, y el AUC de las diez que
        # no se entrenaron se leeria como un resultado del modelo.
        "etiquetas_activas": list(config.etiquetas_activas),
        "determinista": True,
        "learning_rate": config.learning_rate,
        "lote": config.lote,
        "parametros_millones": round(parametros, 2),
        "minutos": round(minutos, 1),
        "mejor_epoca": historial.mejor_epoca,
        "auc_validacion": historial.mejor_auc,
        "epocas": [
            {
                "epoca": e.epoca,
                "perdida_entrenamiento": e.perdida_entrenamiento,
                "perdida_validacion": e.perdida_validacion,
                "auc_macro": e.auc_validacion.macro,
                "auc_por_etiqueta": e.auc_validacion.por_etiqueta,
                "segundos": round(e.segundos, 1),
            }
            for e in historial.epocas
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"guardado en {carpeta}")


if __name__ == "__main__":
    raise SystemExit(main())
