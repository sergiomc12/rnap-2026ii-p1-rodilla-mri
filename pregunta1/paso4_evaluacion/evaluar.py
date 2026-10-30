"""Final evaluation of the trained models on the gold studies.

This is the one time the gold set is used. Every decision -- architecture,
loss, pooling, number of epochs -- was already made against the weak-label
validation split, so what this script reports is generalization and not a
selection effect.

The threshold for precision/recall/F1 is picked on the validation split,
never here: choosing it on the data being reported over would inflate the
numbers deliverable #3 asks for.

Usage:
    python pregunta1/paso4_evaluacion/evaluar.py
    python pregunta1/paso4_evaluacion/evaluar.py --corrida inception_bce_atencion
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Optional, Sequence

import numpy as np

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
from pregunta1.paso3_modelos.modelos import construir_modelo  # noqa: E402
from pregunta1.paso3_modelos.particion import particionar  # noqa: E402
from pregunta1.paso4_evaluacion.metricas import (  # noqa: E402
    auc_macro, bootstrap_auc_macro, mejor_umbral, metricas_en_umbral,
)

SELECCION = _RAIZ / "pregunta1/paso2_preprocesamiento/estudios_seleccionados.csv"
CACHE = _RAIZ / "pregunta1/paso2_preprocesamiento/cache"
TRAIN_CSV = _rutas.TRAIN_CSV
CORRIDAS = _RAIZ / "pregunta1/paso3_modelos/corridas"
SALIDA = _RAIZ / "pregunta1/paso4_evaluacion/resultados.json"


def etiquetas_doradas(ruta: pathlib.Path) -> dict:
    with ruta.open(encoding="utf-8", newline="") as handle:
        reales = {}
        for fila in csv.DictReader(handle):
            valores = [fila[e].strip() for e in LABELS]
            if all(v != "" for v in valores):
                reales[fila["StudyInstanceUID"]] = [int(v) for v in valores]
    return reales


@torch.no_grad()
def predecir(modelo, registros, dispositivo, lote: int = 8):
    """Probabilities and truth for a set of studies."""
    modelo.eval()
    cargador = DataLoader(DatasetRodilla(registros), batch_size=lote)
    probabilidades, verdades = [], []
    for tensores, etiquetas in cargador:
        salida = modelo(tensores.to(dispositivo))
        probabilidades.append(salida.probabilidades().cpu().numpy())
        verdades.append(etiquetas.numpy())
    return np.concatenate(probabilidades), np.concatenate(verdades)


def predecir_corrida(carpeta: pathlib.Path, particion, dispositivo) -> dict:
    """Primera fase: todo lo que necesita la GPU, y nada más.

    Separar esto de las métricas no es organización: es la corrección de un
    fallo. Intercalando inferencia en GPU con los bucles de NumPy del
    bootstrap, el proceso abortaba con una violación de acceso en la tercera
    corrida evaluada --- reproducible con tres corridas cualesquiera, y
    ausente al evaluar una sola, que es por lo que antes se había dado por
    transitorio. Agrupando todo el trabajo de GPU primero y midiendo después,
    con una única transición entre ambos mundos, no vuelve a ocurrir.
    """
    meta = json.loads((carpeta / "historial.json").read_text(encoding="utf-8"))
    modelo = construir_modelo(meta["modelo"], pooling=meta["pooling"],
                              preentrenado=False)
    modelo.load_state_dict(torch.load(carpeta / "pesos.pt", map_location="cpu"))
    modelo = modelo.to(dispositivo)

    p_val, v_val = predecir(modelo, particion.validacion, dispositivo)
    p_oro, v_oro = predecir(modelo, particion.prueba, dispositivo)

    del modelo
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()

    # Copias propias en float64: las métricas no vuelven a tocar nada que
    # administre otra biblioteca.
    return {
        "nombre": carpeta.name, "meta": meta,
        "p_val": np.array(p_val, dtype=np.float64),
        "v_val": np.array(v_val, dtype=np.int64),
        "p_oro": np.array(p_oro, dtype=np.float64),
        "v_oro": np.array(v_oro, dtype=np.int64),
    }


def medir_corrida(predicciones: dict) -> dict:
    """Segunda fase: solo aritmética sobre los arreglos ya calculados."""
    meta = predicciones["meta"]
    p_val, v_val = predicciones["p_val"], predicciones["v_val"]
    p_oro, v_oro = predicciones["p_oro"], predicciones["v_oro"]

    # El umbral se elige acá, en validación, y se aplica sin tocar a los
    # dorados: elegirlo sobre los dorados sería medir sobre los mismos datos
    # con los que se ajustó.
    umbral = mejor_umbral(p_val, v_val)
    auc = auc_macro(p_oro, v_oro)
    intervalo = bootstrap_auc_macro(p_oro, v_oro, remuestreos=2000)
    en_umbral = metricas_en_umbral(p_oro, v_oro, umbral.umbral)

    return {
        "nombre": predicciones["nombre"],
        "modelo": meta["modelo"],
        "perdida": meta["perdida"],
        "pooling": meta["pooling"],
        "parametros_millones": meta["parametros_millones"],
        "auc_validacion": meta["auc_validacion"],
        "auc_dorados": intervalo.valor,
        "ic95": [intervalo.bajo, intervalo.alto],
        "auc_por_etiqueta": auc.por_etiqueta,
        "etiquetas_indefinidas": auc.n_indefinidas,
        "umbral": umbral.umbral,
        "precision": en_umbral.precision,
        "recall": en_umbral.recall,
        "f1": en_umbral.f1,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corridas", type=pathlib.Path, default=CORRIDAS)
    parser.add_argument("--corrida", action="append",
                        help="evaluar solo estas (por nombre de carpeta)")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registros = cargar_registros(SELECCION, CACHE, etiquetas_doradas(TRAIN_CSV))
    particion = particionar(registros)

    carpetas = sorted(c for c in args.corridas.iterdir()
                      if c.is_dir() and (c / "historial.json").is_file())
    if args.corrida:
        carpetas = [c for c in carpetas if c.name in args.corrida]
    if not carpetas:
        print("no hay corridas para evaluar")
        return 1

    print(f"EVALUACION SOBRE {len(particion.prueba)} ESTUDIOS DORADOS")
    print(f"umbral elegido en los {len(particion.validacion)} de validacion\n")

    # Cada corrida se evalúa con el caché con el que se entrenó. Antes todas
    # usaban el sagital, así que la ablación de plano medía un modelo coronal
    # contra imágenes de otro plano y el número no significaba nada. Las
    # corridas viejas no traen el campo y caen al caché por defecto, que para
    # ellas es el correcto.
    por_defecto = (str(SELECCION), str(CACHE))
    particiones = {por_defecto: particion}

    # Fase 1: la GPU. Fase 2: las métricas. Ver predecir_corrida().
    predicciones = []
    for carpeta in carpetas:
        meta = json.loads((carpeta / "historial.json").read_text(encoding="utf-8"))
        clave = (meta.get("seleccion", por_defecto[0]),
                 meta.get("cache", por_defecto[1]))
        if clave not in particiones:
            print(f"  {carpeta.name}: evaluando con {clave[1]}")
            particiones[clave] = particionar(cargar_registros(
                pathlib.Path(clave[0]), pathlib.Path(clave[1]),
                etiquetas_doradas(TRAIN_CSV)))
        predicciones.append(
            predecir_corrida(carpeta, particiones[clave], dispositivo))

    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()
    resultados = [medir_corrida(p) for p in predicciones]
    _imprimir(resultados)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nguardado en {args.salida}")
    return 0


def _imprimir(resultados: list[dict]) -> None:
    print(f"{'corrida':34} {'AUC val':>8} {'AUC oro':>8} {'IC95%':>18} "
          f"{'F1':>6} {'umbral':>7}")
    print("-" * 88)
    for r in resultados:
        ic = f"[{r['ic95'][0]:.3f}, {r['ic95'][1]:.3f}]"
        print(f"{r['nombre']:34} {r['auc_validacion']:>8.3f} {r['auc_dorados']:>8.3f} "
              f"{ic:>18} {r['f1']:>6.3f} {r['umbral']:>7.2f}")

    print(f"\n{'etiqueta':18}", end="")
    for r in resultados:
        print(f"{r['modelo'][:12]:>14}", end="")
    print()
    print("-" * (18 + 14 * len(resultados)))
    for etiqueta in LABELS:
        print(f"{etiqueta:18}", end="")
        for r in resultados:
            valor = r["auc_por_etiqueta"].get(etiqueta)
            print(f"{'n/d' if valor is None else f'{valor:.3f}':>14}", end="")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
