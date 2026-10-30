"""Lee los resultados de la ablación de cortes y dice si la diferencia existe.

El criterio está fijado antes de mirar los números, que es la única forma de que
signifique algo: una configuración mejora si su AUC medio en validación cruzada
supera al de la línea base (8 grupos) **por más de la desviación conjunta entre
pliegues**. Con cinco pliegues sobre 520 estudios esa desviación ronda 0.014, de
modo que la resolución real es de unos 0.03.

Si ninguna supera ese umbral, la conclusión es que ver más cortes no cambia el
resultado medible, no que "mejora un poco".

Usage:
    python pregunta1/paso5_ablacion_gradcam/resumir_ablacion_cortes.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

ABLACION = _RAIZ / "pregunta1/paso5_ablacion_gradcam/ablacion_cortes.json"
CRUZADA = _RAIZ / "pregunta1/paso3_modelos/validacion_cruzada.json"
CORRIDAS = _RAIZ / "pregunta1/paso3_modelos/corridas"
SALIDA = _RAIZ / "pregunta1/paso5_ablacion_gradcam/ablacion_cortes.md"


def _auc_una_particion(nombre: str) -> Optional[float]:
    ruta = CORRIDAS / nombre / "historial.json"
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))["auc_validacion"]


def comparar(base: dict, otra: dict) -> dict:
    """Diferencia entre dos configuraciones, con su escala de ruido.

    La desviación conjunta de dos medias de ``n`` pliegues es
    ``sqrt((s1^2 + s2^2) / n)``. Se compara la diferencia contra dos veces esa
    cantidad, que es el equivalente grueso de un intervalo del 95%.
    """
    delta = otra["auc_media"] - base["auc_media"]
    n = min(base["pliegues"], otra["pliegues"])
    conjunta = math.sqrt((base["auc_desviacion"] ** 2
                          + otra["auc_desviacion"] ** 2) / n)
    return {
        "delta": delta,
        "desviacion_conjunta": conjunta,
        "umbral": 2 * conjunta,
        "supera": abs(delta) > 2 * conjunta,
        "mejora": delta > 2 * conjunta,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablacion", type=pathlib.Path, default=ABLACION)
    parser.add_argument("--cruzada", type=pathlib.Path, default=CRUZADA)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    if not args.ablacion.is_file():
        print(f"falta {args.ablacion}: corré ablacion_cortes.py primero")
        return 1
    ablacion = json.loads(args.ablacion.read_text(encoding="utf-8"))
    cruzada = (json.loads(args.cruzada.read_text(encoding="utf-8"))
               if args.cruzada.is_file() else {})

    # Las filas de arquitectura traen "modelo"; las de solo-caché no.
    filas = {k: v for k, v in ablacion.items() if "modelo" in v}
    if not filas:
        print("todavía no hay corridas entrenadas en el informe")
        return 1

    lineas = ["# Ablación: cuántos cortes ve el modelo", ""]
    for modelo in sorted({v["modelo"] for v in filas.values()}):
        del_modelo = {k: v for k, v in filas.items() if v["modelo"] == modelo}
        base_clave = min(del_modelo, key=lambda k: del_modelo[k]["grupos"])
        base_cv = cruzada.get(base_clave)

        print(f"\n== {modelo} + {del_modelo[base_clave]['pooling']}")
        print(f"{'grupos':>7} {'cortes':>7} {'CV media':>10} {'desv':>7} "
              f"{'delta vs base':>14} {'umbral':>8} {'real?':>6} "
              f"{'1 partición':>12} {'min':>6}")

        lineas += [f"## {modelo} + {del_modelo[base_clave]['pooling']}", "",
                   "| grupos | cortes | CV media | desv. | Δ vs base | umbral "
                   "| ¿real? | 1 partición | caché GB |",
                   "|---:|---:|---:|---:|---:|---:|:--:|---:|---:|"]

        for clave in sorted(del_modelo, key=lambda k: del_modelo[k]["grupos"]):
            fila = del_modelo[clave]
            cv = cruzada.get(clave)
            sola = _auc_una_particion(clave)
            if cv is None:
                print(f"{fila['grupos']:>7} {fila['cortes']:>7} "
                      f"{'(sin CV)':>10}")
                continue

            if clave == base_clave or base_cv is None:
                delta_txt, umbral_txt, marca = "—", "—", "—"
            else:
                c = comparar(base_cv, cv)
                delta_txt = f"{c['delta']:+.4f}"
                umbral_txt = f"±{c['umbral']:.4f}"
                marca = "sí" if c["supera"] else "no"

            print(f"{fila['grupos']:>7} {fila['cortes']:>7} "
                  f"{cv['auc_media']:>10.4f} {cv['auc_desviacion']:>7.4f} "
                  f"{delta_txt:>14} {umbral_txt:>8} {marca:>6} "
                  f"{sola if sola is None else f'{sola:.4f}':>12} "
                  f"{fila.get('minutos_entrenamiento', 0):>6.1f}")

            lineas.append(
                f"| {fila['grupos']} | {fila['cortes']} | "
                f"{cv['auc_media']:.4f} | {cv['auc_desviacion']:.4f} | "
                f"{delta_txt} | {umbral_txt} | {marca} | "
                f"{'—' if sola is None else f'{sola:.4f}'} | "
                f"{fila.get('cache_gb', 0):.1f} |")
        lineas.append("")

    args.salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\nescrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
