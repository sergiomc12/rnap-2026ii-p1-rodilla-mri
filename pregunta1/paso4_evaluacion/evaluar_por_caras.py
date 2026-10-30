"""Evalua en los 58 DORADOS el modelo repartido por caras.

Cada etiqueta se predice con el modelo de la cara que mejor la representa, y
cada cara lee su propio cache: sagital, coronal o axial. Un modelo coronal
evaluado sobre tensores sagitales produce un numero que no significa nada, asi
que la ruta del cache sale de `configuracion.json` de cada corrida y no se
escribe a mano.

El reparto se decidio sobre los 869 estudios de VALIDACION. Este script mide
sobre los 58 DORADOS, que nunca se tocaron: son los unicos con etiquetas de
radiologo y el unico conjunto cuyo numero estima generalizacion. Reportar el
valor de validacion seria reportar el conjunto donde se eligio.

Se corre UNA sola vez. Si despues de ver este resultado se cambia el reparto y
se vuelve a correr, el numero deja de ser una estimacion honesta.

Uso:
    python pregunta1/paso4_evaluacion/evaluar_por_caras.py
"""

from __future__ import annotations

import json
import pathlib
import sys

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

import numpy as np
import torch
from torch.utils.data import DataLoader

from pregunta1 import rutas as _rutas
from pregunta1.paso3_modelos.datos import LABELS, DatasetRodilla, cargar_registros
from pregunta1.paso3_modelos.entrenamiento import cargar_pesos
from pregunta1.paso3_modelos.entrenar import etiquetas_doradas
from pregunta1.paso3_modelos.modelos import construir_modelo
from pregunta1.paso3_modelos.particion import particionar
from pregunta1.paso4_evaluacion.metricas import (
    auc_macro, auc_roc, bootstrap_auc_macro,
)

CORRIDAS = _RAIZ / "pregunta1/paso3_modelos/corridas"
SALIDA = _RAIZ / "pregunta1/paso4_evaluacion/evaluacion_por_caras.json"

#: Que corrida responde por cada etiqueta. El reparto sale de medir cada plano
#: en validacion: `PF OA` va a axial y no a coronal porque ahi gana +0.060 --el
#: corte axial muestra la rotula asentada en el surco femoral, que es donde
#: ocurre la artrosis patelofemoral--, mientras `Lateral Meniscus` se queda en
#: coronal, donde gana +0.033.
RUTEO_RESNET50 = {
    "cara_sagital_rn50_rec": ("Effusion", "MCL", "Contusion", "Lateral OA",
                              "Baker's", "Fracture", "Medial Meniscus",
                              "Medial OA"),
    "cara_coronal_rn50": ("Lateral Meniscus",),
    "cara_axial_rn50": ("Synovitis", "ACL"),
    "cara_axial_rn50_2etiq": ("PF OA",),
}

#: El mismo reparto con inception, para la comparacion entre arquitecturas que
#: pide el entregable. Inception nunca corrio `PF OA` en axial, asi que esa
#: etiqueta se queda en coronal: es el mejor modelo que esa arquitectura tiene
#: para ella, no una eleccion de diseno.
RUTEO_INCEPTION = {
    "cara_sagital_a60": ("Effusion", "MCL", "Contusion", "Lateral OA",
                         "Baker's", "Fracture", "Medial Meniscus", "Medial OA"),
    "cara_coronal_a60": ("Lateral Meniscus", "PF OA"),
    "cara_axial_25ep": ("Synovitis", "ACL"),
}

#: El modelo unico de doce etiquetas, para contrastar contra el reparto. Se
#: usa `doce_etiquetas_36a60` --la continuacion-- y no la corrida original:
#: tiene mejor AUC (0.7220 en su epoca 39 contra 0.7192 en la 35) y, a
#: diferencia de aquella, dejo su `configuracion.json`, asi que su cache y su
#: particion salen del disco en vez de reconstruirse de memoria.
UNICO = {"doce_etiquetas_36a60": LABELS}


def predecir(nombre: str, dispositivo: str):
    """(probabilidades, verdad, uids) de una corrida sobre los dorados."""
    carpeta = CORRIDAS / nombre
    cfg = json.loads((carpeta / "configuracion.json").read_text(encoding="utf-8"))
    registros = cargar_registros(pathlib.Path(cfg["seleccion"]),
                                 pathlib.Path(cfg["cache"]),
                                 etiquetas_doradas(_rutas.TRAIN_CSV))
    particion = particionar(registros, semilla=cfg["semilla"])

    modelo = construir_modelo(cfg["modelo"], pooling=cfg["pooling"],
                              preentrenado=False)
    cargar_pesos(carpeta / "pesos.pt", modelo)
    modelo.to(dispositivo).eval()

    datos = DatasetRodilla(particion.prueba)
    probas, verdades = [], []
    with torch.no_grad():
        for tensores, etiquetas in DataLoader(datos, batch_size=2):
            probas.append(modelo(tensores.to(dispositivo))
                          .probabilidades().cpu().numpy())
            verdades.append(etiquetas.numpy())
    uids = [r.uid for r in particion.prueba]
    print(f"  {nombre:<28} {cfg['modelo']:<9} "
          f"{pathlib.Path(cfg['cache']).name:<20} {len(uids)} dorados")
    return np.concatenate(probas), np.concatenate(verdades), uids


def combinar(ruteo: dict, dispositivo: str):
    """Arma el vector de doce tomando cada etiqueta de su cara."""
    probas = np.zeros((0, 12))
    verdad = np.zeros((0, 12))
    orden = None
    columnas = {}

    for nombre, etiquetas in ruteo.items():
        p, v, uids = predecir(nombre, dispositivo)
        if orden is None:
            orden, probas, verdad = uids, p.copy(), v.copy()
        else:
            # Los dorados salen de la misma particion con la misma semilla, asi
            # que el orden deberia coincidir; si no, la fila 7 de una cara se
            # compararia contra el estudio de la fila 7 de OTRA, y el AUC
            # resultante seria ruido con apariencia de resultado.
            if uids != orden:
                indice = {u: i for i, u in enumerate(uids)}
                p = p[[indice[u] for u in orden]]
        for etiqueta in etiquetas:
            columnas[etiqueta] = p[:, LABELS.index(etiqueta)]

    faltan = [e for e in LABELS if e not in columnas]
    if faltan:
        raise ValueError(f"etiquetas sin cubrir: {faltan}")

    final = np.stack([columnas[e] for e in LABELS], axis=1)
    return final, verdad


def combinar_en(ruteo: dict, dispositivo: str, cual: str):
    """Igual que `combinar` pero sobre el split que se pida.

    El umbral se elige en VALIDACION y se aplica sin tocar a los dorados; por
    eso hace falta predecir sobre los dos, no solo sobre el conjunto que se
    reporta. Elegirlo sobre los dorados inflaria el F1 que se informa.
    """
    import numpy as np
    from pregunta1.paso3_modelos.datos import DatasetRodilla
    from torch.utils.data import DataLoader

    columnas, verdad, orden = {}, None, None
    for nombre, etiquetas in ruteo.items():
        carpeta = CORRIDAS / nombre
        cfg = json.loads((carpeta / "configuracion.json").read_text(encoding="utf-8"))
        registros = cargar_registros(pathlib.Path(cfg["seleccion"]),
                                     pathlib.Path(cfg["cache"]),
                                     etiquetas_doradas(_rutas.TRAIN_CSV))
        particion = particionar(registros, semilla=cfg["semilla"])
        datos = DatasetRodilla(getattr(particion, cual))
        modelo = construir_modelo(cfg["modelo"], pooling=cfg["pooling"],
                                  preentrenado=False)
        cargar_pesos(carpeta / "pesos.pt", modelo)
        modelo.to(dispositivo).eval()
        probas, verdades = [], []
        with torch.no_grad():
            for t, e in DataLoader(datos, batch_size=2):
                probas.append(modelo(t.to(dispositivo)).probabilidades().cpu().numpy())
                verdades.append(e.numpy())
        p = np.concatenate(probas)
        if verdad is None:
            verdad = np.concatenate(verdades)
        for etiqueta in etiquetas:
            columnas[etiqueta] = p[:, LABELS.index(etiqueta)]
    return np.stack([columnas[e] for e in LABELS], axis=1), verdad


def informar(titulo: str, probas, verdad) -> dict:
    r = auc_macro(probas, verdad)
    ic = bootstrap_auc_macro(probas, verdad, remuestreos=2000)
    print()
    print(f"### {titulo}")
    print(f"  macro AUC = {ic}")
    print(f"  {'etiqueta':<18}{'AUC':>9}{'positivos':>11}")
    por_etiqueta = {}
    for i, etiqueta in enumerate(LABELS):
        v = auc_roc(probas[:, i], verdad[:, i])
        pos = int(verdad[:, i].sum())
        por_etiqueta[etiqueta] = v
        celda = f"{v:>9.4f}" if v is not None else f"{'--':>9}"
        print(f"  {etiqueta:<18}{celda}{pos:>11}")
    return {"macro": ic.valor, "ic95": [ic.bajo, ic.alto],
            "por_etiqueta": por_etiqueta,
            "definidas": len(r.definidas)}


def main() -> int:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {dispositivo}\n")
    resultados = {}

    for titulo, ruteo in (("RESNET-50 por caras", RUTEO_RESNET50),
                          ("INCEPTION por caras", RUTEO_INCEPTION),
                          ("INCEPTION modelo unico de 12", UNICO)):
        print(f"--- {titulo} ---")
        try:
            probas, verdad = combinar(ruteo, dispositivo)
        except Exception as e:
            print(f"  no se pudo: {e}")
            continue
        resultados[titulo] = informar(titulo, probas, verdad)
        print()

    SALIDA.write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"guardado en {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
