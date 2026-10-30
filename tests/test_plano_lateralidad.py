"""La normalizacion de lateralidad y el volteo solo valen en sagital.

Dos correcciones del 13-oct-2026, ambas del mismo tipo: una transformacion que
es correcta para cortes sagitales y que corrompe los datos en coronal o axial.

1. `preprocesar_serie` invertia el orden de los cortes en rodillas izquierdas.
   En sagital corresponde --el eje entre cortes es medial-lateral y ese si se
   invierte con el lado-- pero en coronal el eje es anterior-posterior y en
   axial superior-inferior. Medido: el 28% de los estudios son izquierdos, o
   sea que aplicarlo en coronal ponia los cortes posteriores donde van los
   anteriores en uno de cada cuatro.

2. La aumentacion volteaba horizontalmente. Dentro de un corte coronal el eje
   horizontal ES el medial-lateral, asi que voltear intercambia `Medial OA`
   con `Lateral OA` y los dos meniscos.

El mismo error ya habia aparecido en `herramienta_rodilla`, donde corrompia la
etiqueta de 4 de las 6 clases la mitad de las veces.
"""

from __future__ import annotations

import numpy as np
import pytest

from pregunta1.paso2_preprocesamiento.geometria import MetadatosCorte
from pregunta1.paso2_preprocesamiento.preprocesamiento import (
    ConfigPreproceso, preprocesar_serie)
from pregunta1.paso3_modelos.datos import DatasetRodilla, Registro


def _serie_izquierda(n: int = 12):
    """Una serie de rodilla IZQUIERDA con cortes distinguibles entre si.

    La marca es ESPACIAL --una fila brillante distinta en cada corte-- y no un
    valor constante: `_preparar_corte` normaliza la intensidad dentro de cada
    corte, asi que un corte uniforme sale todo en cero y dos cortes distintos
    se vuelven indistinguibles.
    """
    imagenes = []
    for i in range(n):
        c = np.zeros((32, 32), dtype=np.float32)
        c[i * 2, :] = 1.0                  # la fila brillante marca el corte
        imagenes.append(c)
    metadatos = [
        MetadatosCorte(filas=32, columnas=32, espaciado_mm=1.0,
                       posicion=(float(i), 0.0, 0.0),
                       orientacion=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
                       lateralidad="L")
        for i in range(n)
    ]
    return imagenes, metadatos


def _primer_valor(tensor: np.ndarray) -> float:
    return float(tensor[0, 0, 0, 0])


class TestLateralidadPorPlano:
    def test_sagital_invierte_la_rodilla_izquierda(self):
        imagenes, metadatos = _serie_izquierda()
        cfg = ConfigPreproceso(grupos=2, cortes_por_grupo=1, lado=32,
                               plano="Sagittal")
        t = preprocesar_serie(imagenes, metadatos, cfg)
        recto = preprocesar_serie(
            imagenes, metadatos,
            ConfigPreproceso(grupos=2, cortes_por_grupo=1, lado=32,
                             plano="Sagittal", normalizar_lateralidad=False))
        assert not np.allclose(t, recto), (
            "en sagital, una rodilla izquierda DEBE invertirse")

    @pytest.mark.parametrize("plano", ["Coronal", "Axial"])
    def test_coronal_y_axial_no_se_invierten(self, plano):
        """El resultado tiene que ser el mismo con y sin la normalizacion."""
        imagenes, metadatos = _serie_izquierda()
        con = preprocesar_serie(
            imagenes, metadatos,
            ConfigPreproceso(grupos=2, cortes_por_grupo=1, lado=32,
                             plano=plano, normalizar_lateralidad=True))
        sin = preprocesar_serie(
            imagenes, metadatos,
            ConfigPreproceso(grupos=2, cortes_por_grupo=1, lado=32,
                             plano=plano, normalizar_lateralidad=False))
        assert np.allclose(con, sin), (
            f"en {plano} la lateralidad NO debe cambiar el orden: el eje "
            f"entre cortes no es el medial-lateral")

    def test_el_plano_por_defecto_es_sagital(self):
        """Para no cambiar el comportamiento de lo ya medido."""
        assert ConfigPreproceso().plano == "Sagittal"


class TestVolteoPorPlano:
    def _dataset(self, tmp_path, voltear):
        a = np.zeros((2, 1, 4, 4), dtype=np.float32)
        a[..., 0] = 1.0                     # marca asimetrica a la izquierda
        ruta = tmp_path / "x.npy"
        np.save(ruta, a)
        reg = Registro(uid="x", ruta=ruta, etiquetas=(0,) * 12,
                       es_dorado=False, idioma="xx")
        return DatasetRodilla([reg], aumentar=True, semilla=0, voltear=voltear)

    def test_sin_voltear_la_marca_nunca_cambia_de_lado(self, tmp_path):
        d = self._dataset(tmp_path, voltear=False)
        for _ in range(30):
            t = d[0][0].numpy()
            assert t[0, 0, 0, 0] == pytest.approx(1.0), (
                "con voltear=False la imagen no debe espejarse nunca")

    def test_con_voltear_alguna_vez_cambia(self, tmp_path):
        d = self._dataset(tmp_path, voltear=True)
        lados = {d[0][0].numpy()[0, 0, 0, 0] for _ in range(30)}
        assert len(lados) > 1, "con voltear=True debe espejarse a veces"
