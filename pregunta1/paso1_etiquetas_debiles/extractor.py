"""Rule-based weak label extraction from a radiology report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .negacion import EstadoAsercion, clasificar_asercion
from .oraciones import dividir_en_oraciones
from .terminos import LABELS, TERMINOS, Termino


@dataclass(frozen=True)
class Evidencia:
    """One sentence that spoke about one finding, and what it said."""

    etiqueta: str
    oracion: str
    estado: EstadoAsercion
    texto_coincidente: str


@dataclass(frozen=True)
class Extraccion:
    """The twelve labels read off one report, with their evidence."""

    etiquetas: dict[str, int]
    evidencias: tuple[Evidencia, ...] = field(default=())

    def positivos(self) -> frozenset[str]:
        return frozenset(k for k, v in self.etiquetas.items() if v == 1)

    def evidencia_de(self, etiqueta: str) -> tuple[Evidencia, ...]:
        return tuple(e for e in self.evidencias if e.etiqueta == etiqueta)


class ExtractorReglas:
    """Reads the twelve findings from a report, sentence by sentence.

    A finding is positive when at least one sentence asserts it. Sentences
    that negate it do not cancel an assertion made elsewhere: reports often
    state a structure is normal in the body and then describe a lesion in
    the impression.

    Hedged mentions ("possible tear") count as positive only when
    `incierto_es_positivo` is set; they are weak labels either way, and
    which choice is better is an empirical question answered against the
    gold set.
    """

    def __init__(self, incierto_es_positivo: bool = False) -> None:
        self.incierto_es_positivo = incierto_es_positivo

    def extraer(self, reporte: str) -> Extraccion:
        etiquetas = {etiqueta: 0 for etiqueta in LABELS}
        evidencias: list[Evidencia] = []

        for oracion in dividir_en_oraciones(reporte):
            for etiqueta, termino in TERMINOS.items():
                evidencia = self._evaluar(etiqueta, termino, oracion)
                if evidencia is None:
                    continue
                evidencias.append(evidencia)
                if self._cuenta_como_positivo(evidencia.estado):
                    etiquetas[etiqueta] = 1

        return Extraccion(etiquetas=etiquetas, evidencias=tuple(evidencias))

    def _evaluar(
        self, etiqueta: str, termino: Termino, oracion: str
    ) -> Optional[Evidencia]:
        """Return the evidence this sentence gives for one finding, if any."""
        coincidencia = termino.estructura.search(oracion)
        if coincidencia is None:
            return None

        # A structural label needs the pathology named in the same sentence:
        # every report mentions the ACL, most of them to say it is intact.
        patologia = None
        if termino.requiere_patologia:
            patologia = termino.patologia.search(oracion)
            if patologia is None:
                return None

        # La negacion se juzga sobre el HALLAZGO, no sobre la estructura. La
        # estructura casi nunca se niega --la rotula siempre esta ahi-- y la
        # ventana de seis palabras alrededor de ella no llega hasta el "without"
        # que anula la lesion. Medido en los 58 dorados: la oracion "The
        # patellofemoral joint, the patellar and trochlear cartilage appears
        # congruent without chondral lesion" producia `PF OA` = 1 en cuatro de
        # los ocho falsos positivos, porque la ventana se tomaba alrededor de
        # "patellofemoral" --precedido solo por "The"-- en vez de alrededor de
        # "chondral lesion", que es lo que el "without" niega.
        juzgado = patologia if patologia is not None else coincidencia
        # Sobre la patologia se mira SOLO la negacion previa. Una negacion
        # posterior al hallazgo casi siempre lo califica en vez de anularlo:
        # "fracture without displacement", "Baker's cyst without rupture",
        # "meniscal tear without extrusion". Medido en los 58 dorados, tomar
        # tambien la posterior hundia el recall de `Fracture` (0.78 -> 0.39),
        # `Contusion` (0.84 -> 0.53) y `Baker's` (0.75 -> 0.33).
        estado = clasificar_asercion(oracion, *juzgado.span(),
                                     solo_previa=patologia is not None)
        return Evidencia(
            etiqueta=etiqueta,
            oracion=oracion,
            estado=estado,
            texto_coincidente=coincidencia.group(0),
        )

    def _cuenta_como_positivo(self, estado: EstadoAsercion) -> bool:
        if estado is EstadoAsercion.AFIRMADO:
            return True
        return estado is EstadoAsercion.INCIERTO and self.incierto_es_positivo
