"""Choosing which studies to work with.

The exam asks for a stratified subset of 300-600 studies "asegurando
representación de casos positivos para cada uno de los doce hallazgos".
That is not a plain random sample: the rare findings would barely appear.
In the weak labels, Fracture has 182 positives out of 4,407 and Lateral OA
183, so a random draw of 600 would land on roughly 25 of each -- too few to
learn from and far too few to measure.

The sample is built in three passes:

1. Every gold study is taken. They are the clean validation set, so they
   must be present no matter what the weak labels say about them.
2. The twelve findings are filled from rarest to most common, each up to a
   quota. Going rarest-first matters: a study positive for Fracture is
   usually positive for something common too, so the common labels fill up
   for free while the rare ones never would.
3. The remainder is drawn at random from what is left, to keep the subset
   representative of the real distribution rather than only of pathology.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)


@dataclass(frozen=True)
class EstudioEtiquetado:
    """One row of etiquetas_debiles.csv."""

    uid: str
    idioma: str
    etiquetas: dict[str, int]
    n_evidencias: int
    es_dorado: bool
    veredicto: str = "con_hallazgo"

    @property
    def positivos(self) -> frozenset[str]:
        return frozenset(k for k, v in self.etiquetas.items() if v == 1)

    @property
    def ilegible(self) -> bool:
        """True cuando no sabemos qué dice el informe.

        Durante todo el proyecto este criterio fue ``n_evidencias == 0``, con
        el supuesto —escrito en el código— de que un estudio sin evidencia no
        es una rodilla sana sino un informe ilegible. Medido, el supuesto es
        falso en tres de cada cuatro casos: de los 805 estudios no dorados sin
        ninguna etiqueta, 610 afirman explícitamente que alguna estructura es
        normal (ver ``paso1_etiquetas_debiles/normalidad.py``).

        Descartarlos a todos no quitaba ruido: quitaba los ejemplos negativos.
        La muestra quedaba con un 6% de rodillas sanas cuando en el corpus son
        el 15%, y el modelo resultante sobre-predice hallazgos —precisión 0.466
        contra recall 0.744—. Ahora solo se descarta lo que de verdad no se
        pudo leer.
        """
        return self.veredicto == "ilegible"

    @property
    def sano_verificado(self) -> bool:
        """El informe afirma que la rodilla está bien: sirve como negativo."""
        return self.veredicto in ("sano_negado", "sano_afirmado")


@dataclass(frozen=True)
class Muestra:
    """The chosen studies, plus how the choice went."""

    estudios: tuple[EstudioEtiquetado, ...]
    dorados: int
    por_cuota: int
    al_azar: int
    descartados_ilegibles: int = 0
    sanos_verificados: int = 0

    @property
    def uids(self) -> tuple[str, ...]:
        return tuple(e.uid for e in self.estudios)

    def prevalencia(self) -> dict[str, int]:
        cuenta = Counter()
        for estudio in self.estudios:
            cuenta.update(estudio.positivos)
        return {etiqueta: cuenta.get(etiqueta, 0) for etiqueta in LABELS}


class MuestreadorEstratificado:
    """Builds the study subset, guaranteeing positives for every finding."""

    def __init__(
        self,
        objetivo: int = 600,
        cuota_por_etiqueta: int = 40,
        excluir_ilegibles: bool = True,
        incluir_sanos: bool = False,
        semilla: int = 42,
    ) -> None:
        if objetivo < 1:
            raise ValueError("objetivo must be >= 1")
        if cuota_por_etiqueta < 1:
            raise ValueError("cuota_por_etiqueta must be >= 1")
        self.objetivo = objetivo
        self.cuota_por_etiqueta = cuota_por_etiqueta
        self.excluir_ilegibles = excluir_ilegibles
        self.incluir_sanos = incluir_sanos
        self.azar = random.Random(semilla)

    def muestrear(self, estudios: Sequence[EstudioEtiquetado]) -> Muestra:
        elegidos: dict[str, EstudioEtiquetado] = {}

        dorados = [e for e in estudios if e.es_dorado]
        for estudio in dorados:
            elegidos[estudio.uid] = estudio

        candidatos, descartados = self._candidatos(estudios)
        por_cuota = self._llenar_cuotas(candidatos, elegidos)
        al_azar = self._completar_al_azar(candidatos, elegidos)

        return Muestra(
            estudios=tuple(elegidos.values()),
            dorados=len(dorados),
            por_cuota=por_cuota,
            al_azar=al_azar,
            descartados_ilegibles=descartados,
            sanos_verificados=sum(1 for e in elegidos.values()
                                  if e.sano_verificado),
        )

    def _candidatos(
        self, estudios: Sequence[EstudioEtiquetado]
    ) -> tuple[list[EstudioEtiquetado], int]:
        disponibles = [e for e in estudios if not e.es_dorado]
        if not self.excluir_ilegibles:
            return disponibles, 0

        # Las rodillas sanas verificadas quedan FUERA por defecto, y esto se
        # midió en vez de suponerse. Dejarlas entrar parecía lo correcto —son
        # el ejemplo negativo más informativo que hay— y el resultado fue el
        # contrario: AUC sobre los dorados de 0.708 a 0.664, con bootstrap
        # pareado de $-0.045$ IC95% [-0.081, -0.010], significativo.
        #
        # El motivo está medido: de las 43 222 celdas que hoy se entrenan como
        # cero, solo el 6.4% está verificada como negativa; el 93.4% nunca se
        # mencionó en el informe. Un estudio "sano" lo es porque el informe
        # dice que ALGUNA estructura está normal, no porque no haya nada: el
        # único dorado que el módulo clasificó así tiene un hallazgo real. Se
        # agregaban falsos negativos disfrazados de negativos limpios.
        #
        # ``incluir_sanos=True`` reproduce el experimento. La salida correcta
        # no es esa bandera sino tratar las celdas en blanco como desconocidas
        # (política U-Ignore de CheXpert), que es trabajo pendiente.
        legibles = [e for e in disponibles if not e.ilegible]
        utiles = legibles
        if not self.incluir_sanos:
            utiles = [e for e in legibles if not e.sano_verificado]
        # Se devuelven solo los descartados por ilegibles: los sanos no se
        # "descartan" por no poder leerlos sino por una decisión de
        # composición, y mezclarlos en un contador haría ilegible el log.
        return utiles, len(disponibles) - len(legibles)

    def _llenar_cuotas(
        self,
        candidatos: Sequence[EstudioEtiquetado],
        elegidos: dict[str, EstudioEtiquetado],
    ) -> int:
        agregados = 0
        for etiqueta in self._de_mas_rara_a_mas_comun(candidatos):
            faltan = self.cuota_por_etiqueta - sum(
                1 for e in elegidos.values() if etiqueta in e.positivos
            )
            if faltan <= 0:
                continue
            positivos = [e for e in candidatos
                         if etiqueta in e.positivos and e.uid not in elegidos]
            self.azar.shuffle(positivos)
            for estudio in positivos[:faltan]:
                if len(elegidos) >= self.objetivo:
                    return agregados
                elegidos[estudio.uid] = estudio
                agregados += 1
        return agregados

    def _completar_al_azar(
        self,
        candidatos: Sequence[EstudioEtiquetado],
        elegidos: dict[str, EstudioEtiquetado],
    ) -> int:
        restantes = [e for e in candidatos if e.uid not in elegidos]
        self.azar.shuffle(restantes)
        faltan = max(0, self.objetivo - len(elegidos))
        for estudio in restantes[:faltan]:
            elegidos[estudio.uid] = estudio
        return min(faltan, len(restantes))

    @staticmethod
    def _de_mas_rara_a_mas_comun(
        candidatos: Sequence[EstudioEtiquetado],
    ) -> list[str]:
        cuenta = Counter()
        for estudio in candidatos:
            cuenta.update(estudio.positivos)
        return sorted(LABELS, key=lambda etiqueta: cuenta.get(etiqueta, 0))


def cargar_etiquetados(ruta: Path) -> list[EstudioEtiquetado]:
    """Read etiquetas_debiles.csv as `EstudioEtiquetado` records."""
    with Path(ruta).open(encoding="utf-8", newline="") as handle:
        return [
            EstudioEtiquetado(
                uid=fila["StudyInstanceUID"],
                idioma=fila["idioma"],
                etiquetas={e: int(fila[e]) for e in LABELS},
                n_evidencias=int(fila["n_evidencias"]),
                es_dorado=fila["es_dorado"] == "1",
                # Los CSV generados antes de que existiera la columna no la
                # traen; sin veredicto, el estudio se trata como legible.
                veredicto=fila.get("veredicto", "con_hallazgo"),
            )
            for fila in csv.DictReader(handle)
        ]
