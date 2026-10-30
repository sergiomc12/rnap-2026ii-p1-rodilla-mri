"""Genera las tablas del informe a partir de los JSON de las corridas.

Este archivo existe por un error concreto. El informe de la Pregunta 1 se
escribió con los números copiados a mano de una corrida; cuando el pipeline
volvió a entrenar, los JSON cambiaron y el ``.tex`` no, de modo que el PDF
quedó afirmando una ablación significativa que los datos en disco ya no
sostenían. Nadie lo habría notado leyendo el informe.

Desde acá, cada tabla se construye leyendo el JSON que la produjo y el ``.tex``
la incluye con ``\\input``. Si un número del informe está mal, está mal en el
JSON, y eso se puede rastrear hasta la corrida.

Usage:
    python pregunta1/paso6_informe/generar_tablas.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]

ENTRADAS = {
    "resultados": "pregunta1/paso4_evaluacion/resultados.json",
    "comparaciones": "pregunta1/paso4_evaluacion/comparaciones.json",
    "detalle": "pregunta1/paso4_evaluacion/detalle_por_etiqueta.json",
    "validacion_cruzada": "pregunta1/paso3_modelos/validacion_cruzada.json",
    "determinismo": "pregunta1/paso3_modelos/determinismo.json",
    "determinismo_sin_arreglo": "pregunta1/paso3_modelos/determinismo_sin_arreglo.json",
    "patron": "pregunta1/paso5_ablacion_gradcam/patron_gradcam.json",
    "figuras_gradcam": "pregunta1/paso5_ablacion_gradcam/figuras/resumen.json",
}
SALIDA = _RAIZ / "pregunta1/paso6_informe/informe/tablas"

# Las tres configuraciones que el informe compara en la tabla principal.
CONFIGURACIONES = (
    ("inception_bce_atencion", "A --- Inception desde cero"),
    ("resnet50_bce_atencion", "B --- ResNet-50 ImageNet"),
    ("resnet50_promedio", "B + agregación por promedio"),
)

# Nombre clínico en español de cada etiqueta, como las cita el informe.
NOMBRES_ETIQUETA = {
    "ACL": "Lig.\\ cruzado ant.",
    "MCL": "Lig.\\ colateral med.",
    "Medial Meniscus": "Menisco medial",
    "Lateral Meniscus": "Menisco lateral",
    "Medial OA": "Osteoartrosis med.",
    "Lateral OA": "Osteoartrosis lat.",
    "PF OA": "Osteoartrosis PF",
    "Effusion": "Derrame articular",
    "Synovitis": "Sinovitis",
    "Baker's": "Quiste de Baker",
    "Contusion": "Contusión ósea",
    "Fracture": "Fractura",
}


def _num(valor, decimales: int = 3, guion: str = "---") -> str:
    """Formatea un número, o un guión si no está definido.

    El guión no es cosmético: una etiqueta sin positivos tiene AUC indefinida,
    no cero, y la diferencia es la que sostiene media página de la discusión.
    """
    if valor is None:
        return guion
    return f"{valor:.{decimales}f}"


def _ic(bajo, alto, decimales: int = 3, con_signo: bool = False) -> str:
    if bajo is None or alto is None:
        return "---"
    if con_signo:
        return f"$[{bajo:+.{decimales}f}, {alto:+.{decimales}f}]$"
    return f"$[{bajo:.{decimales}f}, {alto:.{decimales}f}]$"


# Nombre legible de cada comparación. Las claves son las descripciones que
# escribe comparar.py; capitalizar la descripción cruda daba cosas como
# "Con vs. sin pesos de imagenet".
NOMBRES_COMPARACION = {
    "entregable #2: las dos arquitecturas": "Inception vs.\\ ResNet-50",
    "ablacion: pooling por media vs. atencion": "Promedio vs.\\ atención",
    "ablacion: con vs. sin pesos de ImageNet": "ImageNet vs.\\ desde cero",
    "ablacion: focal vs. BCE (con atencion)": "Focal vs.\\ BCE (con atención)",
    "ablacion: focal vs. BCE (con media)": "Focal vs.\\ BCE (con promedio)",
    "desbalance: BCE ponderada vs. BCE": "BCE ponderada vs.\\ BCE",
    "ablacion: plano sagital vs. coronal": "Plano sagital vs.\\ coronal",
}


def _signo(valor, decimales: int = 3) -> str:
    return "---" if valor is None else f"${valor:+.{decimales}f}$"


def _tabular(columnas: str, cabecera: Sequence[str],
             filas: Sequence[Sequence[str]], leyenda: str,
             etiqueta: str, extra: Sequence[str] = ()) -> str:
    lineas = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{leyenda}}}",
              f"\\label{{{etiqueta}}}", f"\\begin{{tabular}}{{{columnas}}}",
              "\\toprule", " & ".join(cabecera) + " \\\\", "\\midrule"]
    lineas += [" & ".join(f) + " \\\\" for f in filas]
    lineas += ["\\bottomrule", "\\end{tabular}"]
    lineas += list(extra)
    lineas += ["\\end{table}", ""]
    return "\n".join(lineas)


def _por_nombre(resultados: Sequence[dict]) -> dict[str, dict]:
    return {r["nombre"]: r for r in resultados}


def tabla_principal(resultados: Sequence[dict]) -> str:
    """Entregable #2: las dos arquitecturas más la mejor configuración."""
    indice = _por_nombre(resultados)
    filas, mejor = [], None
    for clave, _ in CONFIGURACIONES:
        fila = indice.get(clave)
        if fila and (mejor is None or fila["auc_dorados"] > mejor):
            mejor = fila["auc_dorados"]

    for clave, titulo in CONFIGURACIONES:
        fila = indice.get(clave)
        if fila is None:
            continue
        auc = _num(fila["auc_dorados"])
        if fila["auc_dorados"] == mejor:
            auc = f"\\textbf{{{auc}}}"
        filas.append([
            titulo, f"{fila['parametros_millones']:.1f}~M", auc,
            _ic(*fila["ic95"]), _num(fila["f1"]), f"{fila['umbral']:.2f}",
        ])
    return _tabular(
        "lrcccc",
        ["Configuración", "Parám.", "AUC macro", "IC95\\%", "$F_1$", "Umbral"],
        filas,
        "Resultados sobre los estudios dorados. El umbral se selecciona sobre "
        "el conjunto de validación y se aplica sin modificación.",
        "tab:principal")


def tabla_ablacion(comparaciones: Sequence[dict]) -> str:
    """El bootstrap pareado, con la significancia marcada tal como salió."""
    filas = []
    for c in comparaciones:
        marca = "\\textbf{sí}" if c["significativa"] else "no"
        nombre = NOMBRES_COMPARACION.get(
            c["descripcion"], c["descripcion"].replace("#", "\\#"))
        filas.append([
            nombre, _signo(c["delta"]),
            _ic(c["bajo"], c["alto"], con_signo=True),
            f"{c['probabilidad_a_mejor']:.0%}".replace("%", "\\%"), marca,
        ])
    significativas = sum(1 for c in comparaciones if c["significativa"])
    return _tabular(
        "lcccc",
        ["Comparación", "$\\Delta$ AUC", "IC95\\%", "$P(A>B)$", "Signif."],
        filas,
        f"Bootstrap pareado sobre los estudios dorados. "
        f"{significativas} de {len(comparaciones)} diferencias alcanzan "
        f"significancia, entendida como un intervalo del 95\\% que no cruza "
        f"cero. El bootstrap es \\emph{{pareado}} —remuestrea los mismos "
        f"estudios para las dos configuraciones— porque comparar dos "
        f"intervalos independientes tiene mucha menos potencia.",
        "tab:ablacion")


def tabla_por_etiqueta(detalle: dict, resultados: Sequence[dict]) -> str:
    """Desglose por hallazgo, con la matriz de confusión de la mejor config."""
    indice = _por_nombre(resultados)
    por_etiqueta_a = (indice.get("inception_bce_atencion") or {}).get(
        "auc_por_etiqueta", {})
    por_etiqueta_b = (indice.get("resnet50_bce_atencion") or {}).get(
        "auc_por_etiqueta", {})

    filas = []
    for d in detalle["etiquetas"]:
        nombre = NOMBRES_ETIQUETA.get(d["etiqueta"], d["etiqueta"])
        intervalo = _ic(d["bajo"], d["alto"], 2)
        if d.get("cruza_azar"):
            intervalo += "$^{*}$"     # su intervalo incluye 0.5
        filas.append([
            nombre, str(d["positivos"]),
            _num(por_etiqueta_a.get(d["etiqueta"])),
            _num(por_etiqueta_b.get(d["etiqueta"])),
            f"{_num(d['auc'])} {intervalo}",
            str(d["tp"]), str(d["fp"]), str(d["fn"]), str(d["tn"]),
            _num(d["f1"]),
        ])

    cruzan = sum(1 for d in detalle["etiquetas"] if d.get("cruza_azar"))
    return _tabular(
        "lrcccrrrrc",
        ["Etiqueta", "Pos.", "A", "B", "Mejor (IC95\\%)", "TP", "FP", "FN",
         "TN", "$F_1$"],
        filas,
        f"Desglose por etiqueta sobre los estudios dorados. Las tres columnas "
        f"de AUC corresponden a Inception (A), ResNet-50 (B) y la mejor "
        f"configuración; el intervalo y la matriz de confusión son de esta "
        f"última, al umbral {detalle['umbral']:.2f} elegido en validación. Un "
        f"asterisco marca los intervalos que incluyen 0.5: son "
        f"{cruzan} de {len(detalle['etiquetas'])}, es decir hallazgos que no "
        f"se distinguen del azar.",
        "tab:etiquetas")


def tabla_validacion_cruzada(cruzada: dict) -> str:
    """Los cinco pliegues, que miden cuánto dependía todo de una sola partición."""
    # Nombres legibles y orden por AUC: la tabla se lee para comparar, no para
    # buscar una corrida por su nombre de carpeta.
    titulos = {
        "inception_bce_atencion": "Inception + atención",
        "resnet50_bce_atencion": "ResNet-50 + atención",
        "resnet50_bce_promedio": "ResNet-50 + promedio",
    }
    filas = []
    for nombre, datos in sorted(cruzada.items(),
                                key=lambda kv: -kv[1]["auc_media"]):
        filas.append([
            titulos.get(nombre, nombre.replace("_", "\\_")),
            str(datos["pliegues"]),
            _num(datos["auc_media"], 4),
            _num(datos["auc_desviacion"], 4),
            f"$[{datos['auc_min']:.4f}, {datos['auc_max']:.4f}]$",
        ])
    return _tabular(
        "lrccc",
        ["Configuración", "Pliegues", "AUC medio", "Desv.", "Rango"],
        filas,
        "Validación cruzada sobre las etiquetas débiles. La desviación entre "
        "pliegues es la escala con la que hay que leer cualquier diferencia "
        "entre configuraciones medida sobre una sola partición.",
        "tab:cruzada")


def tabla_determinismo(con: dict, sin: Optional[dict]) -> str:
    """La verificación de reproducibilidad, que motivó rehacer los resultados."""
    filas = []
    for datos, titulo in ((sin, "Sembrando solo los generadores"),
                          (con, "Con kernels deterministas")):
        if not datos:
            continue
        peor = datos.get("mayor_diferencia") or {}
        filas.append([
            titulo,
            "idénticas" if datos["identicas"] else
            f"época {datos['primera_epoca_que_difiere']}",
            _num(datos["mejor_auc_a"], 4), _num(datos["mejor_auc_b"], 4),
            "0" if datos["identicas"] else f"{peor.get('diferencia', 0):.1e}",
        ])
    return _tabular(
        "lcccc",
        ["Configuración del entorno", "Primera diferencia", "Mejor AUC (1)",
         "Mejor AUC (2)", "Mayor $\\Delta$"],
        filas,
        "Dos entrenamientos de la misma configuración, con la misma semilla y "
        "los mismos datos, en el mismo proceso. Sembrar los generadores no "
        "alcanza en GPU: cuDNN elige kernels en tiempo de ejecución y varios "
        "acumulan en orden no determinista. La diferencia que eso introduce es "
        "del mismo orden que los efectos que el estudio de ablación pretende "
        "medir, de modo que sin fijar los kernels ninguna de esas "
        "comparaciones significa nada.",
        "tab:determinismo")


def tabla_patron_gradcam(patron: dict) -> str:
    """Consistencia de la activación sobre varios casos por hallazgo.

    La tabla va a dos columnas de hallazgos porque son doce filas y una sola
    columna empujaba el informe por encima del límite de páginas.
    """
    filas_datos = sorted(patron["hallazgos"],
                         key=lambda h: (not h["consistente"], h["dispersion"]))
    mitad = (len(filas_datos) + 1) // 2
    filas = []
    for izquierda, derecha in zip(filas_datos[:mitad], filas_datos[mitad:]):
        celdas = []
        for h in (izquierda, derecha):
            nombre = NOMBRES_ETIQUETA.get(h["hallazgo"], h["hallazgo"])
            if h["consistente"]:
                nombre = f"\\textbf{{{nombre}}}"
            celdas += [nombre, _num(h["dispersion"], 3), str(h["en_borde"])]
        filas.append(celdas)

    consistentes = [h["hallazgo"] for h in filas_datos if h["consistente"]]
    return _tabular(
        "lcc@{\\qquad}lcc",
        ["Hallazgo", "Disp.", "Margen", "Hallazgo", "Disp.", "Margen"],
        filas,
        f"Consistencia de la activación sobre {filas_datos[0]['n']} casos por "
        f"hallazgo: dispersión del centro de masa en coordenadas normalizadas y "
        f"número de casos cuya activación cae en el margen de la imagen. Se "
        f"considera consistente (en negrita) una dispersión menor a 0.20 sin "
        f"ningún caso en el margen: lo cumplen {len(consistentes)} de "
        f"{len(filas_datos)}. Lo informativo no es la posición absoluta ---la "
        f"anatomía se desplaza entre estudios--- sino la dispersión entre casos.",
        "tab:patron")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    datos, faltan = {}, []
    for clave, ruta in ENTRADAS.items():
        archivo = _RAIZ / ruta
        if archivo.is_file():
            datos[clave] = json.loads(archivo.read_text(encoding="utf-8"))
        else:
            faltan.append(ruta)

    if faltan:
        print("faltan estos informes (se omiten sus tablas):")
        for ruta in faltan:
            print(f"  {ruta}")

    args.salida.mkdir(parents=True, exist_ok=True)
    escritas = 0

    def escribir(nombre: str, contenido: str) -> None:
        nonlocal escritas
        if not contenido:
            return
        (args.salida / f"{nombre}.tex").write_text(contenido, encoding="utf-8")
        print(f"  {nombre}.tex")
        escritas += 1

    if "resultados" in datos:
        escribir("principal", tabla_principal(datos["resultados"]))
    if "comparaciones" in datos:
        escribir("ablacion", tabla_ablacion(datos["comparaciones"]))
    if "detalle" in datos and "resultados" in datos:
        escribir("etiquetas",
                 tabla_por_etiqueta(datos["detalle"], datos["resultados"]))
    if "validacion_cruzada" in datos:
        escribir("cruzada", tabla_validacion_cruzada(datos["validacion_cruzada"]))
    if "determinismo" in datos:
        escribir("determinismo",
                 tabla_determinismo(datos["determinismo"],
                                    datos.get("determinismo_sin_arreglo")))
    if "patron" in datos:
        escribir("patron", tabla_patron_gradcam(datos["patron"]))

    print(f"\n{escritas} tablas en {args.salida}")
    return 0 if escritas else 1


if __name__ == "__main__":
    raise SystemExit(main())
