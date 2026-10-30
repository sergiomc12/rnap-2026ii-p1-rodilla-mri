"""Intervalos de confianza y matriz de confusión por etiqueta.

El AUC macro resume doce números en uno y oculta dónde falla el modelo. Peor
aún, su intervalo de confianza es mucho más estrecho que el de cualquier
etiqueta individual, porque promediar doce estimaciones ruidosas cancela
parte del ruido. Reportar solo el macro sugiere una precisión que las
etiquetas sueltas no tienen: MCL se estima sobre nueve positivos.

Este script produce, por etiqueta, el AUC con su intervalo bootstrap y la
matriz de confusión al umbral elegido en validación.

Usage:
    python pregunta1/paso4_evaluacion/detalle_por_etiqueta.py
    python pregunta1/paso4_evaluacion/detalle_por_etiqueta.py --corrida inception_bce_atencion
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

import torch  # noqa: E402

from pregunta1.paso3_modelos.datos import LABELS, cargar_registros  # noqa: E402
from pregunta1.paso3_modelos.modelos import construir_modelo  # noqa: E402
from pregunta1.paso3_modelos.particion import particionar  # noqa: E402
from pregunta1.paso4_evaluacion.evaluar import (  # noqa: E402
    CACHE, CORRIDAS, SELECCION, TRAIN_CSV, etiquetas_doradas, predecir,
)
from pregunta1.paso4_evaluacion.metricas import auc_roc, mejor_umbral  # noqa: E402

SALIDA = _RAIZ / "pregunta1/paso4_evaluacion/detalle_por_etiqueta.json"


@dataclass(frozen=True)
class DetalleEtiqueta:
    """Todo lo que se puede decir de una etiqueta sobre el conjunto dorado."""

    etiqueta: str
    positivos: int
    negativos: int
    auc: Optional[float]
    bajo: Optional[float]
    alto: Optional[float]
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def ancho_intervalo(self) -> Optional[float]:
        if self.bajo is None or self.alto is None:
            return None
        return self.alto - self.bajo

    @property
    def precision(self) -> float:
        return self.tp / max(1, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self.tp / max(1, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def cruza_azar(self) -> bool:
        """True cuando el intervalo incluye 0.5: la etiqueta no se distingue del azar."""
        if self.bajo is None or self.alto is None:
            return True
        return self.bajo <= 0.5 <= self.alto


def bootstrap_etiqueta(
    puntajes: np.ndarray, verdad: np.ndarray, remuestreos: int = 2000,
    semilla: int = 42,
) -> tuple[Optional[float], Optional[float]]:
    """Intervalo percentil del AUC de una sola etiqueta.

    Los remuestreos en que la etiqueta queda con una sola clase se descartan:
    ahí el AUC no existe, y contarlos como 0.5 estrecharía el intervalo
    artificialmente hacia el azar.
    """
    azar = random.Random(semilla)
    n = len(verdad)
    valores = []
    for _ in range(remuestreos):
        indices = [azar.randrange(n) for _ in range(n)]
        valor = auc_roc(puntajes[indices], verdad[indices])
        if valor is not None:
            valores.append(valor)

    if len(valores) < remuestreos // 10:
        return None, None      # demasiados remuestreos degenerados para confiar
    valores.sort()
    return valores[int(0.025 * len(valores))], valores[int(0.975 * len(valores))]


def detallar(
    puntajes: np.ndarray, verdad: np.ndarray, umbral: float,
    remuestreos: int = 2000,
) -> list[DetalleEtiqueta]:
    detalles = []
    predicho = puntajes >= umbral
    for i, etiqueta in enumerate(LABELS):
        y, p = verdad[:, i].astype(bool), predicho[:, i]
        bajo, alto = bootstrap_etiqueta(puntajes[:, i], verdad[:, i], remuestreos)
        detalles.append(DetalleEtiqueta(
            etiqueta=etiqueta,
            positivos=int(y.sum()), negativos=int((~y).sum()),
            auc=auc_roc(puntajes[:, i], verdad[:, i]),
            bajo=bajo, alto=alto,
            tp=int((p & y).sum()), fp=int((p & ~y).sum()),
            fn=int((~p & y).sum()), tn=int((~p & ~y).sum()),
        ))
    return detalles


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrida", default="resnet50_promedio")
    parser.add_argument("--remuestreos", type=int, default=2000)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registros = cargar_registros(SELECCION, CACHE, etiquetas_doradas(TRAIN_CSV))
    particion = particionar(registros)

    carpeta = CORRIDAS / args.corrida
    meta = json.loads((carpeta / "historial.json").read_text(encoding="utf-8"))
    modelo = construir_modelo(meta["modelo"], pooling=meta["pooling"],
                              preentrenado=False)
    modelo.load_state_dict(torch.load(carpeta / "pesos.pt", map_location="cpu"))
    modelo = modelo.to(dispositivo)

    # El umbral se elige en validacion, nunca sobre los dorados.
    p_val, v_val = predecir(modelo, particion.validacion, dispositivo)
    umbral = mejor_umbral(p_val, v_val).umbral

    p_oro, v_oro = predecir(modelo, particion.prueba, dispositivo)
    detalles = detallar(p_oro, v_oro, umbral, args.remuestreos)

    print(f"DETALLE POR ETIQUETA · {args.corrida} · {len(particion.prueba)} dorados")
    print(f"umbral {umbral:.2f} elegido en los {len(particion.validacion)} de validacion\n")

    print(f"{'etiqueta':18} {'pos':>4} {'AUC':>6} {'IC95%':>17} {'ancho':>6}  "
          f"{'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'F1':>6}")
    print("-" * 86)
    for d in detalles:
        ic = (f"[{d.bajo:.3f}, {d.alto:.3f}]" if d.bajo is not None else "n/d")
        auc = f"{d.auc:.3f}" if d.auc is not None else "n/d"
        ancho = f"{d.ancho_intervalo:.3f}" if d.ancho_intervalo else "n/d"
        marca = " *" if d.cruza_azar else ""
        print(f"{d.etiqueta:18} {d.positivos:>4} {auc:>6} {ic:>17} {ancho:>6}  "
              f"{d.tp:>3} {d.fp:>3} {d.fn:>3} {d.tn:>3} {d.f1:>6.3f}{marca}")

    indistinguibles = [d.etiqueta for d in detalles if d.cruza_azar]
    anchos = [d.ancho_intervalo for d in detalles if d.ancho_intervalo]
    print("-" * 86)
    print(f"\n* el intervalo incluye 0.5: {len(indistinguibles)} de 12 etiquetas "
          f"no se distinguen del azar")
    print(f"  {', '.join(indistinguibles)}")
    print(f"\nancho medio del intervalo por etiqueta: {sum(anchos)/len(anchos):.3f}")
    print(f"ancho del intervalo del macro          : 0.096")
    print(f"\nEl macro es {sum(anchos)/len(anchos)/0.096:.1f} veces mas estrecho que una")
    print("etiqueta tipica: promediar doce estimaciones ruidosas cancela parte del ruido.")

    args.salida.write_text(json.dumps(
        {"corrida": args.corrida, "umbral": umbral,
         "etiquetas": [d.__dict__ | {"precision": d.precision, "recall": d.recall,
                                     "f1": d.f1, "cruza_azar": d.cruza_azar}
                       for d in detalles]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nguardado en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
