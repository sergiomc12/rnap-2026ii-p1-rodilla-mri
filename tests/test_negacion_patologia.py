"""La negacion se juzga sobre el HALLAZGO, no sobre la estructura anatomica.

Una etiqueta estructural --`PF OA`, `ACL`, los meniscos-- exige dos cosas en la
misma oracion: la estructura (la rotula) y la patologia (la lesion del
cartilago). El extractor clasificaba la asercion mirando las palabras alrededor
de la ESTRUCTURA, y esa casi nunca se niega: la rotula siempre esta ahi. Lo que
se niega es el hallazgo.

El caso que lo destapo aparece literal en cuatro de los ocho falsos positivos de
`PF OA` en los 58 dorados:

    "The patellofemoral joint, the patellar and trochlear cartilage appears
     congruent without chondral lesion."

`without` esta pegado a `chondral lesion`, pero la ventana se tomaba alrededor
de `patellofemoral` --tres palabras despues del comienzo de la oracion-- donde
lo unico que precede es "The". El informe dice que NO hay lesion y la etiqueta
salia 1.
"""

from __future__ import annotations

import pytest

from pregunta1.paso1_etiquetas_debiles.extractor import ExtractorReglas


def _marca(texto: str, etiqueta: str) -> int:
    return ExtractorReglas().extraer(texto).etiquetas[etiqueta]


class TestNegacionTrasLaEstructura:
    """La negacion aparece DESPUES de la estructura, pegada a la patologia."""

    def test_el_caso_real_de_los_dorados(self):
        texto = ("The patellofemoral joint, the patellar and trochlear "
                 "cartilage appears congruent without chondral lesion.")
        assert _marca(texto, "PF OA") == 0, (
            "'without chondral lesion' niega el hallazgo; la etiqueta no puede "
            "salir positiva porque se nombre la rotula")

    def test_shows_no(self):
        texto = "The patella shows no chondral defect."
        assert _marca(texto, "PF OA") == 0

    def test_negative_for(self):
        texto = "The trochlea is negative for chondromalacia."
        assert _marca(texto, "PF OA") == 0

    def test_en_castellano(self):
        texto = "La rotula no presenta condropatia."
        assert _marca(texto, "PF OA") == 0


class TestNoRompeLoQueYaAndaba:
    """La negacion antes de la estructura tiene que seguir funcionando."""

    def test_negacion_antes_de_la_estructura(self):
        texto = "There is no patellar chondromalacia."
        assert _marca(texto, "PF OA") == 0

    def test_afirmado_sigue_afirmado(self):
        texto = ("There is severe cartilage loss at the patella and trochlea "
                 "with subchondral sclerosis.")
        assert _marca(texto, "PF OA") == 1

    def test_afirmado_en_castellano(self):
        texto = "Condropatia rotuliana de grado III con perdida de cartilago."
        assert _marca(texto, "PF OA") == 1

    @pytest.mark.parametrize("etiqueta,texto", [
        ("ACL", "Complete tear of the anterior cruciate ligament."),
        ("Medial Meniscus", "Horizontal tear of the medial meniscus."),
        ("Fracture", "Acute fracture of the tibial plateau."),
    ])
    def test_otras_etiquetas_afirmadas(self, etiqueta, texto):
        assert _marca(texto, etiqueta) == 1

    @pytest.mark.parametrize("etiqueta,texto", [
        ("ACL", "The anterior cruciate ligament is intact without tear."),
        ("Medial Meniscus", "The medial meniscus shows no tear."),
    ])
    def test_otras_etiquetas_negadas(self, etiqueta, texto):
        assert _marca(texto, etiqueta) == 0


class TestEtiquetasSinPatologia:
    """Las que NO exigen patologia siguen juzgandose por su propio termino."""

    def test_synovitis_afirmada(self):
        assert _marca("There is synovial thickening.", "Synovitis") == 1

    def test_synovitis_negada(self):
        assert _marca("There is no synovial thickening.", "Synovitis") == 0
