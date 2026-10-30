"""Estado de las corridas de entrenamiento, contra la corrida de referencia.

Se lee del LOG y no de `historial.json` a proposito: el historial se escribe
recien al terminar, asi que una corrida en curso --o una que se interrumpio,
como la de 60 epocas que murio en la 37-- no tiene historial, pero si tiene
todas sus epocas en el log.

La referencia es `final_inception`: una sola cara (sagital), las DOCE
etiquetas, 64 grupos, que llego a la epoca 37 de 60 con su mejor macro 0.7192
en la 35. Es contra esa corrida que hay que medir el entrenamiento por cara.

Comparar no es restar dos numeros, y este modulo se encarga de no dejar que
lo parezca. Una corrida por cara reporta el macro de SUS etiquetas (dos u
ocho); la referencia reporta el de las doce. El promedio de ocho etiquetas y
el de doce no son la misma cantidad, asi que la unica comparacion honesta
por epoca es contra el macro de la referencia sobre ESE MISMO subconjunto
--posible solo donde quedo el desglose por etiqueta, es decir las epocas 1 a
5, en `plano_sagital_5ep/historial.json`--. Las dos columnas salen con
nombres distintos (`ref(n)` y `ref(12)`) y la segunda avisa que no es
comparable, en vez de ofrecer un numero que invita a restarlo.

El veredicto real necesita las tres caras terminadas: `--combinar` toma de
cada una solo sus etiquetas, arma el vector de doce y saca el macro. ESE es
el numero que se compara contra la referencia.

Uso:
    python pregunta1/paso3_modelos/estado.py cara_
    python pregunta1/paso3_modelos/estado.py cara_ --resumen
    python pregunta1/paso3_modelos/estado.py --combinar
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from typing import NamedTuple, Optional, Sequence

_AQUI = pathlib.Path(__file__).resolve().parent
CORRIDAS = _AQUI / "corridas"

#: La corrida de una sola cara con las doce etiquetas, interrumpida en la 37.
REFERENCIA = "final_inception"

#: Desglose por etiqueta del MISMO modelo de doce, epocas 1-5. Es lo unico
#: que permite la comparacion sobre un subconjunto: la corrida larga se
#: interrumpio sin escribir historial.json, asi que de la epoca 6 en adelante
#: de ella solo queda el macro de las doce.
REFERENCIA_POR_ETIQUETA = CORRIDAS / "plano_sagital_5ep/historial.json"

LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)

# Dos formatos de linea, segun si la corrida entrena un subconjunto:
#   epoca  5/60 | train 0.4171 | val 0.4226 | AUC 0.6654 | 445s *
#   epoca  3/25 | train 0.4019 | val 0.4015 | AUC(8) 0.6647 | AUC(12) 0.5900 | 488s *
LINEA = re.compile(
    r"epoca\s+(?P<epoca>\d+)/(?P<total>\d+)"
    r"\s*\|\s*train\s+(?P<train>[\d.]+)"
    r"\s*\|\s*val\s+(?P<val>[\d.]+)"
    r"\s*\|\s*AUC(?:\((?P<n>\d+)\))?\s+(?P<auc>[\d.]+)"
    r"(?:\s*\|\s*AUC\(12\)\s+(?P<auc12>[\d.]+))?"
    r"\s*\|\s*(?P<seg>[\d.]+)s(?P<mejor>\s*\*)?"
)


class Epoca(NamedTuple):
    epoca: int
    total: int
    train: float
    val: float
    auc: float
    n: Optional[int]
    auc12: Optional[float]
    segundos: float
    es_mejor: bool


def leer(ruta: pathlib.Path) -> list:
    """Las epocas que el log ya tiene escritas."""
    epocas = []
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINEA.search(linea)
        if m:
            epocas.append(Epoca(
                int(m["epoca"]), int(m["total"]), float(m["train"]),
                float(m["val"]), float(m["auc"]),
                int(m["n"]) if m["n"] else None,
                float(m["auc12"]) if m["auc12"] else None,
                float(m["seg"]), bool(m["mejor"]),
            ))
    return epocas


def _macro(valores: Sequence[Optional[float]]) -> Optional[float]:
    definidos = [v for v in valores if v is not None]
    return sum(definidos) / len(definidos) if definidos else None


def referencia_macro_12(nombres=None) -> dict:
    """{epoca: AUC} de la corrida de referencia, o de las que se indiquen.

    Acepta VARIAS corridas y las une por numero de epoca porque una misma
    curva puede estar partida en dos: la cara sagital de inception corrio sus
    epocas 1-25 en `cara_sagital_25ep` y las 26-60 en `cara_sagital_a60`.
    Comparar contra una sola de las dos mostraria media curva y dejaria la
    otra mitad en blanco, como si no existiera.

    Si dos corridas cubren la misma epoca gana la ULTIMA de la lista: al
    reanudar, la fase nueva reemplaza a la vieja en el tramo solapado.
    """
    salida = {}
    for nombre in (nombres or [REFERENCIA]):
        log = CORRIDAS / f"{nombre}.log"
        if log.is_file():
            salida.update({e.epoca: e.auc for e in leer(log)})
    return salida


def referencia_sobre(etiquetas: Sequence[str]) -> dict:
    """{epoca: macro de la referencia sobre ESAS etiquetas}, epocas 1-5.

    Es la unica comparacion legitima por epoca contra una corrida por cara.
    Devuelve vacio si el desglose no esta en disco, en vez de inventar un
    numero a partir del macro de las doce.
    """
    if not (etiquetas and REFERENCIA_POR_ETIQUETA.is_file()):
        return {}
    datos = json.loads(REFERENCIA_POR_ETIQUETA.read_text(encoding="utf-8"))
    salida = {}
    for e in datos["epocas"]:
        por_etiqueta = e["auc_por_etiqueta"]
        valor = _macro([por_etiqueta.get(k) for k in etiquetas])
        if valor is not None:
            salida[e["epoca"]] = valor
    return salida


def etiquetas_activas_de(nombre: str) -> tuple:
    """Las etiquetas que entrena una corrida, de su historial o su progreso."""
    for archivo in ("historial.json", "progreso.json"):
        ruta = CORRIDAS / nombre / archivo
        if ruta.is_file():
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            activas = datos.get("etiquetas_activas")
            if activas:
                return tuple(activas)
    return ()


def _carpeta_de(log: pathlib.Path) -> str:
    """El nombre de carpeta que `--etiqueta` le dio a esa corrida.

    Los logs se llaman `cara_sagital.log` y la carpeta `cara_sagital_25ep`:
    el log lo elige quien lanza, la carpeta la elige `--etiqueta`. Se resuelve
    buscando la carpeta que empiece igual, en vez de hardcodear el sufijo.
    """
    for carpeta in sorted(CORRIDAS.iterdir()):
        if carpeta.is_dir() and carpeta.name.startswith(log.stem):
            return carpeta.name
    return log.stem


def informar(ruta: pathlib.Path, detalle: bool, contra=None) -> None:
    epocas = leer(ruta)
    if not epocas:
        return
    ultima = epocas[-1]
    mejor = max(epocas, key=lambda e: e.auc)
    restan = ultima.total - ultima.epoca

    activas = etiquetas_activas_de(_carpeta_de(ruta))
    if contra:
        # Comparacion explicita contra otra(s) corrida(s): se muestra esa
        # columna y no la referencia de doce etiquetas, que no aplica.
        ref12 = referencia_macro_12(contra)
        refn = {}
    else:
        ref12 = referencia_macro_12() if ruta.stem != REFERENCIA else {}
        refn = referencia_sobre(activas) if ruta.stem != REFERENCIA else {}

    titulo = f"### {ruta.stem}"
    if activas:
        titulo += f"  ({len(activas)} etiquetas: {', '.join(activas)})"
    print()
    print(titulo)
    print(f"  {ultima.epoca}/{ultima.total} epocas | mejor AUC {mejor.auc:.4f} "
          f"en la epoca {mejor.epoca}")
    if restan:
        print(f"  faltan {restan} epocas, ~{restan * ultima.segundos / 60:.0f} min "
              f"({ultima.segundos:.0f}s por epoca)")
    else:
        print("  terminada")

    if not detalle:
        return

    n = len(activas) or mejor.n
    # "nueva"/"vieja" y no "AUC"/"ref": las dos son AUC, y lo que hay que poder
    # leer de un vistazo es cual columna es esta corrida y cual la anterior.
    titulo_auc = f"nueva({n})" if n else "AUC"
    cabecera = f"{'ep':>4}{'train':>9}{'val':>9}{titulo_auc:>11}"
    if n:
        cabecera += f"{'nueva(12)':>11}"
    if refn:
        cabecera += f"{'VIEJA(' + str(n) + ')':>11}"
    if ref12:
        titulo_ref = f"OTRA({n})" if contra and n else "vieja(12)"
        cabecera += f"{titulo_ref:>11}"
    print(f"  {cabecera}{'seg':>7}")

    for e in epocas:
        fila = f"{e.epoca:>4}{e.train:>9.4f}{e.val:>9.4f}{e.auc:>11.4f}"
        if n:
            fila += f"{e.auc12:>11.4f}" if e.auc12 is not None else f"{'--':>11}"
        if refn:
            v = refn.get(e.epoca)
            fila += f"{v:>11.4f}" if v is not None else f"{'--':>11}"
        if ref12:
            v = ref12.get(e.epoca)
            fila += f"{v:>11.4f}" if v is not None else f"{'--':>11}"
        print(f"  {fila}{e.segundos:>7.0f}{' *' if e.es_mejor else ''}")

    if refn or ref12:
        print()
        cuantas = f"{n} etiquetas" if n else "las 12 etiquetas"
        print(f"  nueva(*) = ESTA corrida ({cuantas})")
        if contra:
            print(f"  OTRA(*)  = {' + '.join(contra)}")
            print(f"  Las dos promedian las MISMAS {n} etiquetas, asi que la "
                  f"comparacion por epoca es valida.")
        else:
            print(f"  vieja(*) = {REFERENCIA}: una sola cara con las 12 juntas")
        print(f"  (*) = cuantas etiquetas se promedian")
        # Una continuacion comparte epocas con la corrida que continua, asi
        # que `vieja` son SUS PROPIAS epocas anteriores, no otro modelo.
        # Decir "otra corrida" ahi seria falso.
        solapadas = sorted(set(ref12) & {e.epoca for e in epocas})
        if solapadas and not n:
            print(f"  OJO: las epocas {solapadas[0]}-{solapadas[-1]} se solapan "
                  f"con {REFERENCIA}, que es la corrida que ESTA continua.")
            print(f"       Ahi `vieja` no es otro modelo: es lo que esta misma "
                  f"curva marcaba antes de interrumpirse.")
        if refn and not contra:
            print()
            print(f"  >> La comparacion valida es nueva({n}) contra VIEJA({n}): "
                  f"mismas {n} etiquetas.")
            print(f"     Solo hay dato viejo hasta la epoca {max(refn)}; de ahi "
                  f"en adelante esa corrida se")
            print(f"     interrumpio sin dejar su desglose por etiqueta.")
        if ref12 and n and not contra:
            print(f"  >> nueva(12) y vieja(12) NO se comparan contra nada: "
                  f"promedian etiquetas")
            print(f"     que esta corrida no entrena.")


def combinar(nombres: Sequence[str]) -> int:
    """Junta las caras terminadas en un vector de doce y lo compara.

    De cada cara se toman SOLO sus etiquetas, en su mejor epoca --que es la
    que quedo en pesos.pt--. Tomar el maximo por etiqueta a lo largo de las
    epocas mezclaria checkpoints distintos y daria un macro que ningun
    conjunto de pesos guardado alcanza.
    """
    por_etiqueta, procedencia, faltan = {}, {}, []
    for nombre in nombres:
        ruta = CORRIDAS / nombre / "historial.json"
        if not ruta.is_file():
            faltan.append(nombre)
            continue
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        activas = tuple(datos.get("etiquetas_activas") or LABELS)
        # La mejor epoca puede NO estar en esta corrida: una continuacion
        # arranca sembrada con el mejor de la fase anterior, y si nunca lo
        # supera, `mejor_epoca` apunta a una epoca que esta corrida no corrio.
        # Eso no es un error de datos --significa que la continuacion no
        # aporto nada-- y hay que decirlo, no reventar con StopIteration.
        epoca = next((e for e in datos["epocas"]
                      if e["epoca"] == datos["mejor_epoca"]), None)
        if epoca is None:
            corridas = [e["epoca"] for e in datos["epocas"]]
            print(f"  {nombre}: su mejor epoca ({datos['mejor_epoca']}) es "
                  f"ANTERIOR a esta corrida (corrio "
                  f"{min(corridas)}-{max(corridas)}).")
            print(f"     Ninguna de sus {len(corridas)} epocas la supero; el "
                  f"modelo bueno sigue siendo el de la fase previa.")
            faltan.append(nombre)
            continue
        for etiqueta in activas:
            if etiqueta in por_etiqueta:
                print(f"  OJO: {etiqueta} la reclaman dos caras "
                      f"({procedencia[etiqueta]} y {nombre})")
            por_etiqueta[etiqueta] = epoca["auc_por_etiqueta"].get(etiqueta)
            procedencia[etiqueta] = f"{nombre} ep{epoca['epoca']}"

    print()
    print("### combinado de las caras")
    if faltan:
        print(f"  sin terminar todavia (no tienen historial.json): "
              f"{', '.join(faltan)}")
    sin_cubrir = [e for e in LABELS if e not in por_etiqueta]
    if sin_cubrir:
        print(f"  etiquetas sin cubrir: {', '.join(sin_cubrir)}")

    if not por_etiqueta:
        print("  nada que combinar")
        return 1

    print(f"  {'etiqueta':<18}{'AUC':>9}  de")
    for etiqueta in LABELS:
        if etiqueta in por_etiqueta:
            v = por_etiqueta[etiqueta]
            celda = f"{v:>9.4f}" if v is not None else f"{'--':>9}"
            print(f"  {etiqueta:<18}{celda}  {procedencia[etiqueta]}")

    macro = _macro(list(por_etiqueta.values()))
    print()
    if sin_cubrir or faltan:
        print(f"  macro PARCIAL sobre {len(por_etiqueta)} de 12 = {macro:.4f}")
        print("  no es comparable contra la referencia hasta tener las doce")
        return 0

    ref = referencia_macro_12()
    print(f"  macro combinado (12) = {macro:.4f}")
    if ref:
        tope = max(ref, key=lambda k: ref[k])
        # La comparacion justa es a igual cantidad de epocas.
        epocas_cara = max(int(p.split("ep")[-1]) for p in procedencia.values())
        if epocas_cara in ref:
            d = macro - ref[epocas_cara]
            print(f"  referencia en su epoca {epocas_cara} = "
                  f"{ref[epocas_cara]:.4f}  ({d:+.4f})")
        d = macro - ref[tope]
        print(f"  referencia en su mejor epoca ({tope}) = {ref[tope]:.4f}  "
              f"({d:+.4f})")
        print()
        print("  Esto es validacion (869 estudios, etiquetas debiles). Antes de")
        print("  darlo por bueno hay que medirlo en los 58 dorados, una sola vez.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filtro", nargs="?", default="",
                        help="solo los logs cuyo nombre contenga este texto")
    parser.add_argument("--resumen", action="store_true",
                        help="solo el estado, sin la tabla por epoca")
    parser.add_argument("--contra", metavar="CORRIDA",
                        help="compara epoca a epoca contra otra(s) corrida(s), "
                             "separadas por coma. Varias se unen por numero de "
                             "epoca: una curva partida en dos fases "
                             "(cara_X_25ep,cara_X_a60) se reconstruye entera")
    parser.add_argument("--todas", action="store_true",
                        help="incluir las corridas viejas. Por defecto solo "
                             "salen las de las ultimas 24 horas: la carpeta "
                             "acumula 19 logs y la corrida en curso quedaba "
                             "sepultada al final de la lista")
    parser.add_argument("--horas", type=float, default=24.0,
                        help="con cuantas horas de antiguedad se corta "
                             "(defecto 24)")
    parser.add_argument("--combinar", nargs="*", metavar="CORRIDA",
                        help="junta las caras en un vector de doce y lo "
                             "compara contra la referencia. Sin argumentos, "
                             "usa cara_sagital_25ep/cara_coronal_25ep/"
                             "cara_axial_25ep")
    args = parser.parse_args(argv)

    if args.combinar is not None:
        nombres = args.combinar or ["cara_sagital_25ep", "cara_coronal_25ep",
                                    "cara_axial_25ep"]
        return combinar(nombres)

    logs = [r for r in CORRIDAS.glob("*.log") if args.filtro in r.stem]
    if not logs:
        print(f"sin logs que coincidan con {args.filtro!r} en {CORRIDAS}")
        return 1

    # Un filtro explicito es una peticion explicita: no se le aplica el corte
    # por antiguedad, o pedir una corrida vieja por nombre no devolveria nada.
    if not args.todas and not args.filtro:
        corte = time.time() - args.horas * 3600
        recientes = [r for r in logs if r.stat().st_mtime >= corte]
        ocultas = len(logs) - len(recientes)
        if recientes:
            logs = recientes
            if ocultas:
                print(f"({ocultas} corridas de mas de {args.horas:.0f} h "
                      f"ocultas; --todas para verlas)")

    # Por fecha de modificacion: la corrida activa queda al final.
    contra = ([x.strip() for x in args.contra.split(",") if x.strip()]
              if args.contra else None)
    for ruta in sorted(logs, key=lambda r: r.stat().st_mtime):
        informar(ruta, detalle=not args.resumen, contra=contra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
