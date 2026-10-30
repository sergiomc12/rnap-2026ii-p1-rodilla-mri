"""Acumular gradiente tiene que equivaler a un lote mas grande.

Existe por una medicion: con 192 grupos --los tres planos-- y lote 2, el pico
de VRAM llega a 35.1 GB de 47.8 y el asignador de PyTorch entra en
fragmentacion: 3 409 ms por estudio contra 200 con lote 1, diecisiete veces mas
lento. Acumular procesa de a un estudio y actualiza cada N, con lo que el lote
efectivo es N sin pagar su memoria.

Pero solo sirve si es EQUIVALENTE. Dos detalles lo arruinan en silencio:

  1. No dividir la perdida por N: el gradiente acumulado seria la SUMA y no el
     promedio, y el paso del optimizador saldria N veces mas grande.
  2. Olvidar la cola: si la cantidad de lotes no es multiplo de N, los ultimos
     quedan con gradiente pendiente que nunca se aplica.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class _Lineal(nn.Module):
    """Un modelo minimo con la interfaz que espera el Entrenador."""

    def __init__(self) -> None:
        super().__init__()
        self.capa = nn.Linear(4, 3, bias=False)
        nn.init.ones_(self.capa.weight)

    def forward(self, x):
        from pregunta1.paso3_modelos.modelos import SalidaModelo
        return SalidaModelo(logits=self.capa(x), atencion=None)


def _gradiente(lote: int, acumular: int, datos, objetivo):
    """El gradiente acumulado tras una pasada por todos los datos."""
    torch.manual_seed(0)
    m = _Lineal()
    crit = nn.BCEWithLogitsLoss()
    n = len(datos)
    m.zero_grad(set_to_none=True)
    for i in range(0, n, lote):
        x, y = datos[i:i + lote], objetivo[i:i + lote]
        p = crit(m(x).logits, y)
        (p / acumular).backward()
    return m.capa.weight.grad.clone()


class TestEquivalencia:
    def test_acumular_iguala_a_un_lote_grande(self):
        """8 muestras: un lote de 8, contra 4 lotes de 2 acumulados x4."""
        torch.manual_seed(1)
        datos = torch.randn(8, 4)
        objetivo = (torch.rand(8, 3) > 0.5).float()

        grande = _gradiente(lote=8, acumular=1, datos=datos, objetivo=objetivo)
        acumulado = _gradiente(lote=2, acumular=4, datos=datos, objetivo=objetivo)

        assert torch.allclose(grande, acumulado, atol=1e-6), (
            "acumular 4 lotes de 2 debe dar el mismo gradiente que uno de 8")

    def test_sin_dividir_el_gradiente_queda_escalado(self):
        """El error que la division evita: el gradiente sale N veces mayor."""
        torch.manual_seed(1)
        datos = torch.randn(8, 4)
        objetivo = (torch.rand(8, 3) > 0.5).float()

        correcto = _gradiente(lote=2, acumular=4, datos=datos, objetivo=objetivo)
        sin_dividir = _gradiente(lote=2, acumular=1, datos=datos, objetivo=objetivo)

        assert torch.allclose(sin_dividir, correcto * 4, atol=1e-5), (
            "sin dividir por N el gradiente queda N veces mas grande")


class TestConfiguracion:
    def test_por_defecto_no_acumula(self):
        """Para no cambiar el comportamiento de lo ya medido."""
        from pregunta1.paso3_modelos.entrenamiento import ConfigEntrenamiento
        assert ConfigEntrenamiento().acumular == 1

    @pytest.mark.parametrize("n", [1, 2, 4, 8])
    def test_acepta_valores_validos(self, n):
        from pregunta1.paso3_modelos.entrenamiento import ConfigEntrenamiento
        assert ConfigEntrenamiento(acumular=n).acumular == n
