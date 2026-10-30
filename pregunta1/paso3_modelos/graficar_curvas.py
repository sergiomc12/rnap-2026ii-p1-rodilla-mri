"""Curvas de entrenamiento de las corridas, para la sección de experimentos.

Dos paneles por figura: las pérdidas de entrenamiento y validación, y el AUC
macro de validación. El primero muestra el sobreajuste; el segundo, de dónde
sale la época que la detención temprana conserva.

Usage:
    python pregunta1/paso3_modelos/graficar_curvas.py
    python pregunta1/paso3_modelos/graficar_curvas.py --corridas inception_bce_atencion resnet50_bce_atencion
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

CORRIDAS = _RAIZ / "pregunta1/paso3_modelos/corridas"
SALIDA = _RAIZ / "pregunta1/paso3_modelos/figuras"

# Las dos arquitecturas del entregable #2, mas la configuracion ganadora.
POR_DEFECTO = ("inception_bce_atencion", "resnet50_bce_atencion", "resnet50_promedio")

ETIQUETAS = {
    "inception_bce_atencion": "A — Inception desde cero",
    "resnet50_bce_atencion": "B — ResNet-50 ImageNet",
    "resnet50_promedio": "B + pooling por promedio",
}


def cargar(nombre: str) -> Optional[dict]:
    ruta = CORRIDAS / nombre / "historial.json"
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def graficar(nombres: Sequence[str], salida: pathlib.Path) -> int:
    import matplotlib
    matplotlib.use("Agg")      # sin display: esto corre en consola
    import matplotlib.pyplot as plt

    historiales = [(n, cargar(n)) for n in nombres]
    historiales = [(n, h) for n, h in historiales if h is not None]
    if not historiales:
        print("no hay corridas para graficar")
        return 1

    salida.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4))
    colores = plt.cm.tab10.colors

    for indice, (nombre, historial) in enumerate(historiales):
        color = colores[indice % len(colores)]
        etiqueta = ETIQUETAS.get(nombre, nombre)
        epocas = [e["epoca"] for e in historial["epocas"]]

        ejes[0].plot(epocas, [e["perdida_entrenamiento"] for e in historial["epocas"]],
                     color=color, linestyle="-", label=f"{etiqueta} (train)")
        ejes[0].plot(epocas, [e["perdida_validacion"] for e in historial["epocas"]],
                     color=color, linestyle="--", alpha=0.7,
                     label=f"{etiqueta} (val)")

        ejes[1].plot(epocas, [e["auc_macro"] for e in historial["epocas"]],
                     color=color, label=etiqueta)
        # La epoca que la deteccion temprana conserva.
        ejes[1].scatter([historial["mejor_epoca"]], [historial["auc_validacion"]],
                        color=color, marker="o", s=55, zorder=5,
                        edgecolors="white", linewidths=1.2)

    ejes[0].set_xlabel("Época"); ejes[0].set_ylabel("Pérdida")
    ejes[0].set_title("Pérdida de entrenamiento y validación")
    ejes[0].legend(fontsize=7, loc="upper right")
    ejes[0].grid(alpha=0.25)

    ejes[1].set_xlabel("Época"); ejes[1].set_ylabel("AUC macro")
    ejes[1].set_title("AUC de validación (el punto marca la época conservada)")
    ejes[1].legend(fontsize=8, loc="lower right")
    ejes[1].grid(alpha=0.25)

    figura.tight_layout()
    destino = salida / "curvas_entrenamiento.png"
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)

    print(f"{len(historiales)} corridas -> {destino}")
    for nombre, historial in historiales:
        ultima = historial["epocas"][-1]
        print(f"  {nombre:28} mejor epoca {historial['mejor_epoca']:>2} | "
              f"AUC {historial['auc_validacion']:.4f} | "
              f"brecha final train/val {ultima['perdida_validacion']/ultima['perdida_entrenamiento']:.1f}x")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corridas", nargs="*", default=list(POR_DEFECTO))
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)
    return graficar(args.corridas, args.salida)


if __name__ == "__main__":
    raise SystemExit(main())
