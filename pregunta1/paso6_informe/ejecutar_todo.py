"""Run the whole pipeline end to end, with one command.

The exam requires it: "El código debe ejecutarse de extremo a extremo con
un único comando documentado en el README".

Every stage is skipped when its artifact already exists, so a re-run only
does what is missing. `--rehacer` forces everything.

Usage:
    python pregunta1/paso6_informe/ejecutar_todo.py
    python pregunta1/paso6_informe/ejecutar_todo.py --rapido
    python pregunta1/paso6_informe/ejecutar_todo.py --desde 3
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
PYTHON = sys.executable


@dataclass(frozen=True)
class Etapa:
    """One stage of the pipeline and how to tell it already ran."""

    paso: int
    nombre: str
    script: str
    argumentos: tuple[str, ...] = ()
    artefacto: Optional[str] = None

    @property
    def ya_corrio(self) -> bool:
        if self.artefacto is None:
            return False
        ruta = _RAIZ / self.artefacto
        if ruta.is_dir():
            return any(ruta.iterdir())
        return ruta.is_file()


# Translation is the one stage that cannot be made fast: it runs a 600M
# model over 2,671 reports. Everything else is minutes.
ETAPAS = (
    Etapa(1, "Traducir reportes no-ingleses",
          "pregunta1/paso1_etiquetas_debiles/traducir_reportes.py",
          artefacto="pregunta1/paso1_etiquetas_debiles/reportes_traducidos.csv"),
    Etapa(1, "Extraer etiquetas debiles",
          "pregunta1/paso1_etiquetas_debiles/generar_etiquetas.py",
          artefacto="pregunta1/paso1_etiquetas_debiles/etiquetas_debiles.csv"),
    Etapa(1, "Medir el extractor contra los dorados",
          "pregunta1/paso1_etiquetas_debiles/evaluar_extractor.py"),
    Etapa(2, "Elegir la muestra estratificada",
          "pregunta1/paso2_preprocesamiento/seleccionar_estudios.py",
          artefacto="pregunta1/paso2_preprocesamiento/estudios_seleccionados.csv"),
    Etapa(2, "Preprocesar y cachear los tensores",
          "pregunta1/paso2_preprocesamiento/generar_cache.py",
          artefacto="pregunta1/paso2_preprocesamiento/cache"),
    Etapa(3, "Entrenar A: Inception desde cero",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "inception", "--epocas", "25"),
          "pregunta1/paso3_modelos/corridas/inception_bce_atencion"),
    Etapa(3, "Entrenar B: ResNet-50 ImageNet",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "resnet50", "--epocas", "25"),
          "pregunta1/paso3_modelos/corridas/resnet50_bce_atencion"),
    Etapa(5, "Ablacion: pooling por media",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "resnet50", "--pooling", "promedio", "--epocas", "25",
           "--etiqueta", "resnet50_promedio"),
          "pregunta1/paso3_modelos/corridas/resnet50_promedio"),
    Etapa(5, "Ablacion: perdida focal",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "resnet50", "--perdida", "focal", "--epocas", "25",
           "--etiqueta", "resnet50_focal"),
          "pregunta1/paso3_modelos/corridas/resnet50_focal"),
    Etapa(5, "Ablacion: sin pesos de ImageNet",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "resnet50", "--sin-preentrenar", "--epocas", "25",
           "--etiqueta", "resnet50_sin_imagenet"),
          "pregunta1/paso3_modelos/corridas/resnet50_sin_imagenet"),
    Etapa(6, "Desbalance: BCE ponderada",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "resnet50", "--pooling", "promedio",
           "--perdida", "bce_ponderada", "--epocas", "25",
           "--etiqueta", "resnet50_promedio_ponderada"),
          "pregunta1/paso3_modelos/corridas/resnet50_promedio_ponderada"),
    Etapa(2, "Muestra coronal, para la ablacion de plano",
          "pregunta1/paso2_preprocesamiento/seleccionar_estudios.py",
          ("--plano", "Coronal",
           "--salida", "pregunta1/paso2_preprocesamiento/estudios_coronal.csv"),
          "pregunta1/paso2_preprocesamiento/estudios_coronal.csv"),
    Etapa(2, "Cache coronal",
          "pregunta1/paso2_preprocesamiento/generar_cache.py",
          ("--seleccion", "pregunta1/paso2_preprocesamiento/estudios_coronal.csv",
           "--cache", "pregunta1/paso2_preprocesamiento/cache_coronal"),
          "pregunta1/paso2_preprocesamiento/cache_coronal"),
    Etapa(5, "Ablacion: plano coronal",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "resnet50", "--pooling", "promedio", "--epocas", "25",
           "--etiqueta", "resnet50_promedio_coronal",
           "--seleccion", "pregunta1/paso2_preprocesamiento/estudios_coronal.csv",
           "--cache", "pregunta1/paso2_preprocesamiento/cache_coronal"),
          "pregunta1/paso3_modelos/corridas/resnet50_promedio_coronal"),
    Etapa(5, "Ablacion: focal con pooling por promedio",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "resnet50", "--pooling", "promedio", "--perdida", "focal",
           "--epocas", "25", "--etiqueta", "resnet50_promedio_focal"),
          "pregunta1/paso3_modelos/corridas/resnet50_promedio_focal"),
    Etapa(5, "Ablacion: focal con Inception",
          "pregunta1/paso3_modelos/entrenar.py",
          ("--modelo", "inception", "--perdida", "focal", "--epocas", "25",
           "--etiqueta", "inception_focal"),
          "pregunta1/paso3_modelos/corridas/inception_focal"),
    Etapa(4, "Evaluar sobre los dorados",
          "pregunta1/paso4_evaluacion/evaluar.py",
          artefacto="pregunta1/paso4_evaluacion/resultados.json"),
    Etapa(4, "Bootstrap pareado entre configuraciones",
          "pregunta1/paso4_evaluacion/comparar.py",
          artefacto="pregunta1/paso4_evaluacion/comparaciones.json"),
    Etapa(4, "Detalle e intervalos por etiqueta",
          "pregunta1/paso4_evaluacion/detalle_por_etiqueta.py",
          artefacto="pregunta1/paso4_evaluacion/detalle_por_etiqueta.json"),
    # Las tres configuraciones que el informe compara en la tabla de validacion
    # cruzada. El JSON se acumula en vez de pisarse, pero el artefacto solo
    # puede apuntar a una: las otras dos se declaran sin artefacto y se saltan
    # por ser baratas comparadas con el costo de una tabla incompleta.
    Etapa(3, "Validacion cruzada: ResNet-50 + promedio",
          "pregunta1/paso3_modelos/validacion_cruzada.py",
          ("--modelo", "resnet50", "--pooling", "promedio", "--epocas", "25"),
          "pregunta1/paso3_modelos/validacion_cruzada.json"),
    Etapa(3, "Validacion cruzada: Inception",
          "pregunta1/paso3_modelos/validacion_cruzada.py",
          ("--modelo", "inception", "--epocas", "25")),
    Etapa(3, "Validacion cruzada: ResNet-50 + atencion",
          "pregunta1/paso3_modelos/validacion_cruzada.py",
          ("--modelo", "resnet50", "--pooling", "atencion", "--epocas", "25")),
    Etapa(3, "Curvas de entrenamiento",
          "pregunta1/paso3_modelos/graficar_curvas.py",
          artefacto="pregunta1/paso3_modelos/figuras"),
    Etapa(5, "Figuras de Grad-CAM",
          "pregunta1/paso5_ablacion_gradcam/generar_figuras.py",
          artefacto="pregunta1/paso5_ablacion_gradcam/figuras"),
    Etapa(5, "Patron de Grad-CAM sobre varios casos",
          "pregunta1/paso5_ablacion_gradcam/patron_gradcam.py",
          artefacto="pregunta1/paso5_ablacion_gradcam/patron_gradcam.json"),
    # Las tablas SIEMPRE se regeneran: son la traduccion de los JSON a LaTeX.
    # Un informe con numeros de una corrida anterior es invisible al lector, y
    # ya paso una vez en este proyecto.
    Etapa(6, "Generar las tablas del informe desde los JSON",
          "pregunta1/paso6_informe/generar_tablas.py"),
    # Estas dos tampoco declaran artefacto, por lo mismo: la etapa anterior
    # siempre reescribe las tablas, asi que un PDF o un Word que ya existan
    # son justamente los que hay que rehacer.
    Etapa(6, "Compilar el informe NeurIPS (PDF) y verificar el formato",
          "pregunta1/paso6_informe/compilar_informe.py"),
    Etapa(6, "Exportar el informe a Word",
          "pregunta1/paso6_informe/exportar_word.py"),
)


def correr(etapa: Etapa, rapido: bool) -> int:
    orden = [PYTHON, str(_RAIZ / etapa.script), *etapa.argumentos]
    if rapido and "--epocas" in orden:
        orden[orden.index("--epocas") + 1] = "3"
    return subprocess.run(orden, cwd=_RAIZ).returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehacer", action="store_true",
                        help="rehacer aunque el artefacto exista")
    parser.add_argument("--rapido", action="store_true",
                        help="3 epocas por modelo, para verificar el circuito")
    parser.add_argument("--desde", type=int, default=1, help="empezar en este paso")
    parser.add_argument("--hasta", type=int, default=6)
    parser.add_argument("--listar", action="store_true",
                        help="mostrar las etapas y su estado, sin correr nada")
    args = parser.parse_args(argv)

    etapas = [e for e in ETAPAS if args.desde <= e.paso <= args.hasta]

    if args.listar:
        print(f"{'paso':>5}  {'estado':<12} etapa")
        print("-" * 78)
        for e in etapas:
            estado = "ya corrió" if e.ya_corrio else "pendiente"
            print(f"{e.paso:>5}  {estado:<12} {e.nombre}")
        return 0
    print(f"PIPELINE COMPLETO · {len(etapas)} etapas\n")

    comenzado = time.time()
    for indice, etapa in enumerate(etapas, start=1):
        encabezado = f"[{indice}/{len(etapas)}] paso {etapa.paso} — {etapa.nombre}"
        if etapa.ya_corrio and not args.rehacer:
            print(f"{encabezado}: ya existe, se salta")
            continue

        print(f"\n{'=' * 72}\n{encabezado}\n{'=' * 72}", flush=True)
        codigo = correr(etapa, args.rapido)
        if codigo != 0:
            print(f"\nFALLO en: {etapa.nombre} (codigo {codigo})")
            return codigo

    print(f"\n{'=' * 72}")
    print(f"PIPELINE COMPLETO en {(time.time() - comenzado) / 60:.1f} min")
    print("resultados: pregunta1/paso4_evaluacion/resultados.json")
    print("figuras   : pregunta1/paso5_ablacion_gradcam/figuras/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
