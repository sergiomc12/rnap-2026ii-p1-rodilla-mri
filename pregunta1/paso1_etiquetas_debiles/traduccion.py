"""Translating non-English reports to English with NLLB-200.

The rule-based extractor only works in English and Spanish: measured macro
F1 is 0.66 in English and 0.48 in Spanish, but 0.00-0.08 in the other seven
languages, leaving 46.8% of studies with no label at all.

Extending the term dictionary to those seven would not fix it, because the
negation detector works on separate words inside a window. Turkish negates
with a suffix ("yirtilmamis" = not torn), Greek and Bulgarian add a
different script on top. Translating the full text turns nine negation
problems into the one that is already solved and measured.

Reports run up to 4,743 characters, past what the model handles in one
pass, so translation happens sentence by sentence and the sentences are
rejoined. That also preserves the structure the extractor relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .oraciones import dividir_en_oraciones

MODELO = "facebook/nllb-200-distilled-600M"
IDIOMA_DESTINO = "eng_Latn"

# langdetect's two-letter codes mapped to the codes NLLB expects.
CODIGOS_NLLB = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "tr": "tur_Latn",
    "hr": "hrv_Latn",
    "el": "ell_Grek",
    "de": "deu_Latn",
    "bg": "bul_Cyrl",
    "nl": "nld_Latn",
    "fr": "fra_Latn",
}

LOTE = 32
MAXIMO_TOKENS = 256


@dataclass(frozen=True)
class Traduccion:
    """One report translated, with what it took."""

    texto: str
    idioma_origen: str
    n_oraciones: int

    @property
    def fue_traducido(self) -> bool:
        return self.idioma_origen != "en"


class TraductorNLLB:
    """Translates reports into English, one sentence at a time.

    The model is loaded once and reused: loading it per report would
    dominate the runtime.
    """

    def __init__(
        self,
        modelo: str = MODELO,
        dispositivo: Optional[str] = None,
        lote: int = LOTE,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lote = lote
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.modelo = AutoModelForSeq2SeqLM.from_pretrained(modelo).to(self.dispositivo)
        self.modelo.eval()

    def traducir_reporte(self, texto: str, idioma: str) -> Traduccion:
        """Translate one report, leaving English untouched."""
        oraciones = dividir_en_oraciones(texto)
        if idioma == "en" or not oraciones:
            return Traduccion(texto=texto, idioma_origen=idioma,
                              n_oraciones=len(oraciones))

        traducidas = self.traducir_oraciones(oraciones, idioma)
        return Traduccion(texto=" ".join(traducidas), idioma_origen=idioma,
                          n_oraciones=len(oraciones))

    def traducir_oraciones(self, oraciones: Sequence[str], idioma: str) -> list[str]:
        """Translate a batch of sentences from one source language."""
        codigo = CODIGOS_NLLB.get(idioma)
        if codigo is None:
            # An unknown language is left as-is rather than mistranslated
            # under a guessed source code.
            return list(oraciones)

        self.tokenizer.src_lang = codigo
        salida: list[str] = []
        for grupo in _en_lotes(oraciones, self.lote):
            salida.extend(self._traducir_lote(grupo))
        return salida

    def _traducir_lote(self, oraciones: Sequence[str]) -> list[str]:
        entradas = self.tokenizer(
            list(oraciones), return_tensors="pt", padding=True,
            truncation=True, max_length=MAXIMO_TOKENS,
        ).to(self.dispositivo)

        with self._torch.no_grad():
            generados = self.modelo.generate(
                **entradas,
                forced_bos_token_id=self._id_destino(),
                max_new_tokens=MAXIMO_TOKENS,
                num_beams=1,  # greedy: beams cost time without helping terminology
            )
        return self.tokenizer.batch_decode(generados, skip_special_tokens=True)

    def _id_destino(self) -> int:
        """Token id that forces the decoder to emit English."""
        convertir = self.tokenizer.convert_tokens_to_ids
        identificador = convertir(IDIOMA_DESTINO)
        if identificador is not None and identificador >= 0:
            return identificador
        # Older tokenizer builds expose the mapping under a different name.
        return self.tokenizer.lang_code_to_id[IDIOMA_DESTINO]


def _en_lotes(elementos: Sequence[str], tamano: int) -> Iterable[Sequence[str]]:
    for inicio in range(0, len(elementos), tamano):
        yield elementos[inicio:inicio + tamano]
