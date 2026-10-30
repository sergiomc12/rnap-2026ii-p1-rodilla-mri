"""Produce the Grad-CAM figures for deliverable #5.

For each requested finding, picks a gold study where a radiologist marked
it positive, runs Grad-CAM on it, and writes the slice, the heat map and
the overlay side by side.

Usage:
    python pregunta1/paso5_ablacion_gradcam/generar_figuras.py
    python pregunta1/paso5_ablacion_gradcam/generar_figuras.py --corrida resnet50_promedio
"""

from __future__ import annotations

import argparse
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

from pregunta1.paso3_modelos.datos import LABELS, cargar_registros  # noqa: E402
from pregunta1.paso3_modelos.modelos import construir_modelo  # noqa: E402
from pregunta1.paso3_modelos.particion import particionar  # noqa: E402
from pregunta1.paso4_evaluacion.evaluar import etiquetas_doradas  # noqa: E402
from pregunta1.paso5_ablacion_gradcam.gradcam import GradCAM, superponer  # noqa: E402

SELECCION = _RAIZ / "pregunta1/paso2_preprocesamiento/estudios_seleccionados.csv"
CACHE = _RAIZ / "pregunta1/paso2_preprocesamiento/cache"
TRAIN_CSV = _rutas.TRAIN_CSV
CORRIDAS = _RAIZ / "pregunta1/paso3_modelos/corridas"
FIGURAS = _RAIZ / "pregunta1/paso5_ablacion_gradcam/figuras"

# Three findings with distinct anatomy, so the maps can be judged against
# where each one actually lives: effusion fills the joint capsule, the
# medial meniscus sits between femur and tibia, the ACL runs through the
# intercondylar notch.
HALLAZGOS = ("Effusion", "Medial Meniscus", "ACL")


def cargar_modelo(carpeta: pathlib.Path):
    meta = json.loads((carpeta / "historial.json").read_text(encoding="utf-8"))
    modelo = construir_modelo(meta["modelo"], pooling=meta["pooling"],
                              preentrenado=False)
    modelo.load_state_dict(torch.load(carpeta / "pesos.pt", map_location="cpu"))
    return modelo, meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrida", default="resnet50_promedio")
    parser.add_argument("--hallazgos", nargs="*", default=list(HALLAZGOS))
    parser.add_argument("--salida", type=pathlib.Path, default=FIGURAS)
    args = parser.parse_args(argv)

    import cv2

    modelo, meta = cargar_modelo(CORRIDAS / args.corrida)
    registros = cargar_registros(SELECCION, CACHE, etiquetas_doradas(TRAIN_CSV))
    dorados = particionar(registros).prueba
    args.salida.mkdir(parents=True, exist_ok=True)

    print(f"corrida={args.corrida} modelo={meta['modelo']} pooling={meta['pooling']}")
    print(f"{len(dorados)} estudios dorados disponibles\n")

    filas = []
    with GradCAM(modelo) as camara:
        for hallazgo in args.hallazgos:
            indice = LABELS.index(hallazgo)
            casos = [r for r in dorados if r.etiquetas[indice] == 1]
            if not casos:
                print(f"{hallazgo}: sin casos positivos entre los dorados")
                continue

            caso = casos[0]
            tensor = torch.from_numpy(np.load(caso.ruta))
            mapa, atencion, probabilidad = camara.generar(tensor, indice)

            grupo = int(np.argmax(atencion))
            corte = tensor[grupo, 1].numpy()      # the middle slice of the group
            panel = np.hstack([
                cv2.cvtColor((np.clip(corte, 0, 1) * 255).astype(np.uint8),
                             cv2.COLOR_GRAY2BGR),
                cv2.applyColorMap((mapa * 255).astype(np.uint8), cv2.COLORMAP_JET),
                superponer(corte, mapa),
            ])
            destino = args.salida / f"gradcam_{hallazgo.replace(' ', '_')}.png"
            cv2.imwrite(str(destino), panel)

            masa = float((mapa > 0.5).mean())
            print(f"{hallazgo:18} p={probabilidad:.3f} grupo={grupo} "
                  f"area_caliente={masa:.1%} -> {destino.name}")
            filas.append({
                "hallazgo": hallazgo,
                "estudio": caso.uid,
                "probabilidad": probabilidad,
                "grupo_atendido": grupo,
                "atencion_maxima": float(atencion.max()),
                "atencion_uniforme": bool(np.allclose(atencion, atencion[0])),
                "fraccion_area_caliente": masa,
                "figura": destino.name,
            })

    (args.salida / "resumen.json").write_text(
        json.dumps({"corrida": args.corrida, "pooling": meta["pooling"],
                    "casos": filas}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{len(filas)} figuras en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
