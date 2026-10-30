"""Reanudar una corrida sin perder lo entrenado.

Existe por un caso real: la corrida de doce etiquetas llego a la epoca 37 de
60 y se interrumpio, y `entrenar.py` no sabia continuar --relanzarla tiraba 35
epocas de entrenamiento, siete horas de computo--.

Dos detalles lo arruinan en silencio:

  1. Cargar solo los pesos y no el estado del optimizador. AdamW lleva dos
     momentos por parametro; reiniciarlos en cero tras 35 epocas produce un
     bajon transitorio que parece "el modelo empeoro al continuar".
  2. Volver a numerar las epocas desde 1. Las dos mitades de la misma curva
     dejan de poder pegarse, y la epoca 3 de la continuacion se confunde con
     la 3 original, que es un modelo completamente distinto.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from pregunta1.paso3_modelos.entrenamiento import (  # noqa: E402
    ConfigEntrenamiento, cargar_pesos,
)


class _Modelito(nn.Module):
    def __init__(self, valor: float = 0.0) -> None:
        super().__init__()
        self.capa = nn.Linear(3, 2)
        with torch.no_grad():
            self.capa.weight.fill_(valor)
            self.capa.bias.fill_(valor)

    def forward(self, x):
        return self.capa(x)


def _optimizador(modelo):
    return torch.optim.AdamW(modelo.parameters(), lr=1e-3)


class TestFormatoNuevo:
    def test_restaura_pesos_y_optimizador(self, tmp_path):
        origen = _Modelito(0.5)
        opt = _optimizador(origen)
        # Un paso real, para que el optimizador tenga momentos que restaurar.
        origen(torch.randn(4, 3)).sum().backward()
        opt.step()

        ruta = tmp_path / "pesos.pt"
        torch.save({"modelo": origen.state_dict(),
                    "optimizador": opt.state_dict(), "epoca": 35}, ruta)

        destino = _Modelito(0.0)
        opt2 = _optimizador(destino)
        info = cargar_pesos(ruta, destino, opt2)

        assert info["con_optimizador"] is True
        assert info["epoca"] == 35
        assert torch.allclose(destino.capa.weight, origen.capa.weight)
        assert opt2.state_dict()["state"], "los momentos deben quedar cargados"

    def test_sin_pasar_optimizador_igual_carga_los_pesos(self, tmp_path):
        origen = _Modelito(0.7)
        ruta = tmp_path / "pesos.pt"
        torch.save({"modelo": origen.state_dict(), "optimizador": None,
                    "epoca": 12}, ruta)

        destino = _Modelito(0.0)
        info = cargar_pesos(ruta, destino)
        assert torch.allclose(destino.capa.weight, origen.capa.weight)
        assert info["epoca"] == 12


class TestFormatoViejo:
    """Los checkpoints anteriores son un state_dict pelado y deben seguir."""

    def test_carga_un_state_dict_pelado(self, tmp_path):
        origen = _Modelito(0.3)
        ruta = tmp_path / "pesos.pt"
        torch.save(origen.state_dict(), ruta)

        destino = _Modelito(0.0)
        info = cargar_pesos(ruta, destino, _optimizador(destino))

        assert torch.allclose(destino.capa.weight, origen.capa.weight)
        assert info["con_optimizador"] is False, (
            "debe avisar que no hay momentos, no fingir que los restauro")
        assert info["epoca"] is None

    def test_el_formato_se_detecta_por_la_forma(self, tmp_path):
        """No por la fecha ni por una bandera: por si trae la clave 'modelo'."""
        viejo, nuevo = tmp_path / "v.pt", tmp_path / "n.pt"
        m = _Modelito(0.1)
        torch.save(m.state_dict(), viejo)
        torch.save({"modelo": m.state_dict(), "optimizador": None}, nuevo)

        assert cargar_pesos(viejo, _Modelito())["con_optimizador"] is False
        assert cargar_pesos(nuevo, _Modelito())["con_optimizador"] is True


class TestNumeracionDeEpocas:
    def test_por_defecto_arranca_en_uno(self):
        assert ConfigEntrenamiento().epoca_inicial == 1

    def test_acepta_una_epoca_inicial(self):
        assert ConfigEntrenamiento(epoca_inicial=36).epoca_inicial == 36

    def test_rechaza_cero_y_negativos(self):
        with pytest.raises(ValueError, match="epoca_inicial"):
            ConfigEntrenamiento(epoca_inicial=0)
        with pytest.raises(ValueError, match="epoca_inicial"):
            ConfigEntrenamiento(epoca_inicial=-5)

    @pytest.mark.parametrize("inicial,cuantas,esperado", [
        (1, 5, [1, 2, 3, 4, 5]),
        (36, 3, [36, 37, 38]),
        (36, 25, list(range(36, 61))),
    ])
    def test_el_rango_continua_la_corrida_anterior(self, inicial, cuantas,
                                                   esperado):
        """36 + 25 epocas tiene que llegar justo a la 60, no a la 61."""
        cfg = ConfigEntrenamiento(epoca_inicial=inicial, epocas=cuantas)
        primera = cfg.epoca_inicial
        assert list(range(primera, primera + cfg.epocas)) == esperado
        assert esperado[-1] == inicial + cuantas - 1
