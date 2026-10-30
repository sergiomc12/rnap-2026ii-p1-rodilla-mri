"""El lector de estado tiene que entender los DOS formatos de linea.

Existe porque el log es la unica fuente del progreso de una corrida en curso:
`historial.json` se escribe recien al terminar, y la corrida de 60 epocas que
murio en la 37 nunca lo escribio. Si el regex no matchea, el script no dice
"no entiendo esta linea", dice "0 epocas" --indistinguible de una corrida que
todavia no arranco--.
"""

from __future__ import annotations

import pytest

from pregunta1.paso3_modelos.estado import LINEA, leer


VIEJO = "  epoca  5/60 | train 0.4171 | val 0.4226 | AUC 0.6654 | 445s *"
NUEVO = ("  epoca  3/25 | train 0.4019 | val 0.4015 | AUC(8) 0.6647 | "
         "AUC(12) 0.5900 | 488s *")
SIN_MEJORA = "  epoca  4/25 | train 0.4186 | val 0.4234 | AUC 0.6555 | 439s"


class TestFormatoViejo:
    """Doce etiquetas: un solo AUC, sin parentesis."""

    def test_matchea(self):
        assert LINEA.search(VIEJO) is not None

    def test_campos(self):
        m = LINEA.search(VIEJO)
        assert (int(m["epoca"]), int(m["total"])) == (5, 60)
        assert float(m["train"]) == pytest.approx(0.4171)
        assert float(m["val"]) == pytest.approx(0.4226)
        assert float(m["auc"]) == pytest.approx(0.6654)
        assert float(m["seg"]) == pytest.approx(445)

    def test_sin_subconjunto_no_hay_n_ni_auc12(self):
        m = LINEA.search(VIEJO)
        assert m["n"] is None
        assert m["auc12"] is None


class TestFormatoPorCara:
    """Subconjunto: AUC(n) decide, AUC(12) es diagnostico."""

    def test_matchea(self):
        assert LINEA.search(NUEVO) is not None

    def test_el_auc_capturado_es_el_de_las_activas(self):
        """Lo importante: NO debe capturar el AUC(12) como si fuera el bueno."""
        m = LINEA.search(NUEVO)
        assert int(m["n"]) == 8
        assert float(m["auc"]) == pytest.approx(0.6647)
        assert float(m["auc12"]) == pytest.approx(0.5900)

    def test_no_confunde_los_segundos_con_el_auc12(self):
        m = LINEA.search(NUEVO)
        assert float(m["seg"]) == pytest.approx(488)


class TestMarcaDeMejora:
    def test_el_asterisco_marca_la_mejora(self):
        assert LINEA.search(VIEJO)["mejor"] is not None
        assert LINEA.search(NUEVO)["mejor"] is not None

    def test_sin_asterisco_no_es_mejora(self):
        assert LINEA.search(SIN_MEJORA)["mejor"] is None


class TestLeerArchivo:
    def test_mezcla_de_lineas(self, tmp_path):
        log = tmp_path / "x.log"
        log.write_text("4407 estudios | entrenamiento=3480\n"
                       "dispositivo: cuda\n"
                       f"{NUEVO}\n{SIN_MEJORA}\n", encoding="utf-8")
        epocas = leer(log)
        assert [e.epoca for e in epocas] == [3, 4]
        assert epocas[0].es_mejor is True
        assert epocas[1].es_mejor is False
        assert epocas[0].n == 8
        assert epocas[1].n is None

    def test_log_vacio(self, tmp_path):
        log = tmp_path / "vacio.log"
        log.write_text("", encoding="utf-8")
        assert leer(log) == []

    def test_log_sin_epocas_todavia(self, tmp_path):
        """Una corrida que arranco pero no cerro su primera epoca."""
        log = tmp_path / "nuevo.log"
        log.write_text("4407 estudios\nmodelo=inception\n", encoding="utf-8")
        assert leer(log) == []

    def test_no_explota_con_bytes_raros(self, tmp_path):
        """Los logs se leen con errors=replace; un byte suelto no debe cortar."""
        log = tmp_path / "raro.log"
        log.write_bytes(b"\xff\xfe basura\n" + NUEVO.encode("utf-8") + b"\n")
        assert len(leer(log)) == 1
