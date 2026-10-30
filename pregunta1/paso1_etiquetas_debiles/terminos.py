"""Términos de las 12 etiquetas objetivo.

RECONSTRUIDO desde bytecode (el .py se corrompió a bytes nulos por una
colisión de escritura en Windows). Patrones idénticos al original; el
refuerzo de OA (sigla "OA", inner/outer, tricompartmental) está aplicado.
Refuerzo adicional gold-driven de etiquetas débiles: ACL ("VKB" alemán,
"cross-link"), Effusion ("joint discharge", "amount of fluid", "fluid in the
joint"), Lateral Meniscus (coordinación "menisco interno/medial y lateral").

Dos clases de etiqueta: estructurales (ACL, MCL, meniscos, 3 OA) necesitan
estructura + patología en la misma oración; hallazgos directos (effusion,
synovitis, Baker's, contusion, fracture) el término ya es la patología.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
)


@dataclass(frozen=True)
class Termino:
    """Cómo reconocer un hallazgo en una oración.

    Con `patologia`, ambos patrones deben matchear la misma oración; sin
    ella, matchear `estructura` sola es el hallazgo.
    """
    estructura: "re.Pattern"
    patologia: "Optional[re.Pattern]" = None

    @property
    def requiere_patologia(self) -> bool:
        return self.patologia is not None


def _re(patron: str) -> "re.Pattern":
    """Compila un patrón insensible a mayúsculas (como el original)."""
    return re.compile(patron, re.IGNORECASE)


_LESION_LIGAMENTO = _re('\\btear\\b|\\btorn\\b|\\brupture|\\bruptur|\\bsprain|\\bdisrupt|\\binjur|\\bpartial thickness\\b|\\bfull thickness\\b|\\bdeficien|rotura|ruptura|desgarro|lesi[oó]n|esguince|rotos?\\b')
_LESION_MENISCO = _re('\\btear\\b|\\btorn\\b|\\brupture|\\bmaceration\\b|\\bdegenerat|\\bextrusion\\b|\\bbucket.handle\\b|\\bdisplaced fragment\\b|\\bamputa|\\bresect|\\bpostoperative\\b|rotura|ruptura|desgarro|degenerat|amputaci|macerac|lesi[oó]n')
_LESION_CARTILAGO = _re('\\bosteoarthrit|\\barthrosis\\b|\\bchondral\\b|\\bchondropath|\\bchondromalacia\\b|\\bcartilage (loss|defect|thinning|damage)\\b|\\bjoint space narrowing\\b|\\bosteophyt|\\bsubchondral (sclerosis|cyst)\\b|\\bdegenerative change|artrosis|osteoartrosis|gonartrosis|condropat|condromalacia|p[eé]rdida de cart[ií]lago|osteofit|pinzamiento|esclerosis subcondral|\\bgonorrh|\\bchondromalati|\\bartery cartilage\\b|\\bcartilage (baldness|disorders?)\\b|\\bOA\\b|\\bchondros[ei]s\\b|\\bchondrotic\\b')

TERMINOS: "dict[str, Termino]" = {
    'ACL': Termino(estructura=_re('\\bACL\\b|\\banterior cruciate\\b|\\bLCA\\b|ligamento cruzado anterior|cruzado anterior|\\bVKB\\b|cross.?link'), patologia=_LESION_LIGAMENTO),
    'MCL': Termino(estructura=_re('\\bMCL\\b|\\bmedial collateral\\b|\\btibial collateral\\b|\\bLCM\\b|ligamento colateral (medial|interno)|colateral medial'), patologia=_LESION_LIGAMENTO),
    'Medial Meniscus': Termino(estructura=_re('\\bmedial meniscus\\b|\\b(?:internal|inner) meniscus\\b|\\bmeniscus internus\\b|menisco (medial|interno)'), patologia=_LESION_MENISCO),
    'Lateral Meniscus': Termino(estructura=_re('\\blateral meniscus\\b|\\b(?:external|outer) meniscus\\b|\\bmeniscus externus\\b|menisco (lateral|externo)|menisco (?:interno|medial) y lateral'), patologia=_LESION_MENISCO),
    'Medial OA': Termino(estructura=_re('\\bmedial (compartment|femorotibial|tibiofemoral)\\b|compartimento (medial|interno)|femorotibial (medial|interno)|\\binner (compartment|femoro\\w*|tibio\\w*|femoral|tibial|condyl\\w*)\\b|\\b(?:tricompartmental|all (?:three )?compartments?)\\b|tricompartimental|(?:los |las )?tres compartiment\\w*'), patologia=_LESION_CARTILAGO),
    'Lateral OA': Termino(estructura=_re('\\blateral (compartment|femorotibial|tibiofemoral)\\b|compartimento (lateral|externo)|femorotibial (lateral|externo)|\\bouter (compartment|femoro\\w*|tibio\\w*|femoral|tibial|condyl\\w*)\\b|\\b(?:tricompartmental|all (?:three )?compartments?)\\b|tricompartimental|(?:los |las )?tres compartiment\\w*'), patologia=_LESION_CARTILAGO),
    'PF OA': Termino(estructura=_re('\\b(?:retro|peri|para|infra|supra)?patell(?:a|ar|ofemoral)\\b|\\btrochlea|(?:retro|peri|para)?patelofemoral|(?:retro)?rotulian|r[oó]tula|tr[oó]clea|\\b(?:tricompartmental|all (?:three )?compartments?)\\b|tricompartimental|(?:los |las )?tres compartiment\\w*'), patologia=_LESION_CARTILAGO),
    'Effusion': Termino(estructura=_re('\\beffusion\\b|\\bjoint fluid\\b|\\bsuprapatellar (fluid|distension)\\b|derrame( articular)?|l[ií]quido articular|efusi[oó]n|\\bsuprapatell\\w*\\s+(recess\\w*|pouch|buildup)\\b|\\bsuperpigmental\\b|\\bhydrops\\b|\\bintra.articular discharge\\b|\\bjoint discharge\\b|\\bfluid in the (knee )?joint\\b|\\bamount of fluid\\b')),
    'Synovitis': Termino(estructura=_re('\\bsynovit|\\bsynovial (thickening|proliferation|hypertroph)\\b|\\bthickened synovial\\b|\\bpannus\\b|sinovit|engrosamiento sinovial|hipertrofia sinovial')),
    "Baker's": Termino(estructura=_re("\\bbaker'?s? cyst\\b|\\bpopliteal cyst\\b|\\bgastrocnemio.semimembranosus\\b|quiste de baker|quiste popl[ií]teo|\\bpopulated cyst\\b|\\bpoplite\\w* (cyst|bladder)\\b|\\bbaker'?s? \\w+ (bladder|cyst)\\b")),
    'Contusion': Termino(estructura=_re('\\bbone (contusion|bruise|marrow edema|marrow oedema)\\b|\\bcontusion\\b|\\bbone marrow signal (change|alteration)\\b|\\btrabecular edema\\b|contusi[oó]n [oó]sea|edema (de m[eé]dula|[oó]seo|medular)|\\bosteomyell\\w*|\\bbone marrow swelling\\b|\\bmedullary o?edema\\b')),
    'Fracture': Termino(estructura=_re('\\bfracture\\b|\\bfractured\\b|\\bavulsion\\b|\\bsegond\\b|fractura|arrancamiento [oó]seo|\\bfractu\\s+hour\\b')),
}

assert set(TERMINOS) == set(LABELS), "every label needs a term entry"
