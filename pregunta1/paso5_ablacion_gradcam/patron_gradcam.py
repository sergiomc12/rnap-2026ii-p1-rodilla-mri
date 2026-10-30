"""Grad-CAM sobre varios casos por hallazgo, para distinguir patrón de casualidad.

Con un solo ejemplo por hallazgo no se puede decir si el modelo mira
sistemáticamente la región correcta o si acertó una vez. Este script corre
Grad-CAM sobre todos los casos dorados positivos de cada hallazgo y resume
dónde cae la activación.

La posición se mide como el centro de masa del mapa, en coordenadas
normalizadas: (0,0) es la esquina superior izquierda y (1,1) la inferior
derecha. Lo que importa no es el valor absoluto ---la anatomía se desplaza
entre estudios--- sino la **dispersión** entre casos: una desviación chica
indica que el modelo mira siempre el mismo lugar.

Usage:
    python pregunta1/paso5_ablacion_gradcam/patron_gradcam.py
    python pregunta1/paso5_ablacion_gradcam/patron_gradcam.py --max-casos 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

import torch  # noqa: E402

from pregunta1.paso3_modelos.datos import LABELS, cargar_registros  # noqa: E402
from pregunta1.paso3_modelos.particion import particionar  # noqa: E402
from pregunta1.paso4_evaluacion.evaluar import (  # noqa: E402
    CACHE, CORRIDAS, SELECCION, TRAIN_CSV, etiquetas_doradas,
)
from pregunta1.paso5_ablacion_gradcam.generar_figuras import cargar_modelo  # noqa: E402
from pregunta1.paso5_ablacion_gradcam.gradcam import GradCAM  # noqa: E402

SALIDA = _RAIZ / "pregunta1/paso5_ablacion_gradcam/patron_gradcam.json"

# Umbral sobre el mapa normalizado a partir del cual se considera que la
# activacion "esta ahi". 0.5 es la mitad del maximo de cada mapa.
UMBRAL_CALIENTE = 0.5
BORDE = 0.15   # fraccion de la imagen que se considera margen


@dataclass(frozen=True)
class CasoMapa:
    """Un estudio, su probabilidad y dónde cayó la activación."""

    estudio: str
    probabilidad: float
    centro_x: float
    centro_y: float
    fraccion_caliente: float

    @property
    def en_el_borde(self) -> bool:
        """True cuando el centro de masa cae en el margen de la imagen.

        Una activación en el borde no puede corresponder a la articulación:
        el preprocesamiento recorta 130 mm centrados en la rodilla.
        """
        return not (BORDE < self.centro_x < 1 - BORDE
                    and BORDE < self.centro_y < 1 - BORDE)


@dataclass(frozen=True)
class PatronHallazgo:
    """Resumen de todos los casos de un hallazgo."""

    hallazgo: str
    casos: tuple[CasoMapa, ...]

    @property
    def n(self) -> int:
        return len(self.casos)

    @property
    def dispersion(self) -> Optional[float]:
        """Desviación del centro de masa entre casos: cuán consistente es."""
        if self.n < 2:
            return None
        dx = statistics.stdev(c.centro_x for c in self.casos)
        dy = statistics.stdev(c.centro_y for c in self.casos)
        return (dx ** 2 + dy ** 2) ** 0.5

    @property
    def en_borde(self) -> int:
        return sum(1 for c in self.casos if c.en_el_borde)

    @property
    def probabilidad_media(self) -> float:
        return statistics.mean(c.probabilidad for c in self.casos) if self.n else 0.0

    @property
    def consistente(self) -> bool:
        """Mira el mismo sitio en todos los casos y nunca el borde."""
        return (self.dispersion is not None and self.dispersion < 0.20
                and self.en_borde == 0)


def centro_de_masa(mapa: np.ndarray) -> tuple[float, float, float]:
    """Centro de masa de la región caliente, normalizado a [0,1]."""
    caliente = mapa >= UMBRAL_CALIENTE
    fraccion = float(caliente.mean())
    if not caliente.any():
        return 0.5, 0.5, 0.0
    ys, xs = np.nonzero(caliente)
    alto, ancho = mapa.shape
    return float(xs.mean() / ancho), float(ys.mean() / alto), fraccion


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrida", default="resnet50_promedio")
    parser.add_argument("--max-casos", type=int, default=8)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, meta = cargar_modelo(CORRIDAS / args.corrida)
    modelo = modelo.to(dispositivo)

    registros = cargar_registros(SELECCION, CACHE, etiquetas_doradas(TRAIN_CSV))
    dorados = particionar(registros).prueba

    print(f"PATRON DE GRAD-CAM · {args.corrida} · hasta {args.max_casos} casos "
          f"por hallazgo\n")
    print(f"{'hallazgo':18} {'n':>3} {'p media':>8} {'centro':>14} "
          f"{'dispersion':>11} {'borde':>6}")
    print("-" * 70)

    patrones = []
    with GradCAM(modelo) as camara:
        for indice, hallazgo in enumerate(LABELS):
            positivos = [r for r in dorados if r.etiquetas[indice] == 1]
            casos = []
            for registro in positivos[:args.max_casos]:
                tensor = torch.from_numpy(np.load(registro.ruta)).to(dispositivo)
                mapa, _, probabilidad = camara.generar(tensor, indice)
                cx, cy, fraccion = centro_de_masa(mapa)
                casos.append(CasoMapa(registro.uid, probabilidad, cx, cy, fraccion))

            patron = PatronHallazgo(hallazgo, tuple(casos))
            patrones.append(patron)

            if patron.n == 0:
                print(f"{hallazgo:18} {0:>3}  (sin casos positivos entre los dorados)")
                continue

            cx = statistics.mean(c.centro_x for c in casos)
            cy = statistics.mean(c.centro_y for c in casos)
            disp = f"{patron.dispersion:.3f}" if patron.dispersion else "n/d"
            marca = " *" if patron.consistente else ""
            print(f"{hallazgo:18} {patron.n:>3} {patron.probabilidad_media:>8.3f} "
                  f"({cx:.2f}, {cy:.2f})".ljust(48)
                  + f"{disp:>11} {patron.en_borde:>6}{marca}")

    consistentes = [p.hallazgo for p in patrones if p.consistente]
    print("-" * 70)
    print(f"\n* activacion consistente (dispersion < 0.20 y ningun caso en el borde):")
    print(f"  {len(consistentes)} de 12 -> {', '.join(consistentes) or 'ninguno'}")

    con_borde = [(p.hallazgo, p.en_borde, p.n) for p in patrones if p.en_borde]
    if con_borde:
        print(f"\nhallazgos con casos en el borde de la imagen (fuera de la articulacion):")
        for nombre, borde, n in con_borde:
            print(f"  {nombre:18} {borde}/{n} casos")

    args.salida.write_text(json.dumps({
        "corrida": args.corrida,
        "umbral_caliente": UMBRAL_CALIENTE,
        "hallazgos": [{
            "hallazgo": p.hallazgo, "n": p.n,
            "probabilidad_media": p.probabilidad_media,
            "dispersion": p.dispersion, "en_borde": p.en_borde,
            "consistente": p.consistente,
            "casos": [c.__dict__ | {"en_el_borde": c.en_el_borde} for c in p.casos],
        } for p in patrones],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nguardado en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
