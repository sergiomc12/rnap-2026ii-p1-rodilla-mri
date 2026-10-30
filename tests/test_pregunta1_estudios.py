"""`pregunta1/estudios.py`: el lector de `train.csv` que vive dentro de la pregunta.

**Por qué este archivo existe, y por qué el último test es el importante.** Este
módulo es una copia deliberada de ``src/rsna/selection.py``. La copia se hizo
para que ``pregunta1/`` se pueda **compartir sola**: con el import original
(``sys.path[:0] = [..., "src"]``) la carpeta no arrancaba del otro lado
(``ModuleNotFoundError: rsna``). La otra copia **no** es código muerto —
``src/rsna/cli.py``, la herramienta de descarga, la sigue usando — así que las
dos van a existir en paralelo.

Dos copias de la misma lógica divergen. ``test_las_dos_copias_coinciden``
compara las dos **sobre el CSV real** para que, si alguien toca una, el test
falle en vez de que el proyecto reporte dos verdades distintas.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from pregunta1.estudios import (
    LABELS, Series, Study, gold_studies, load_series, load_studies,
)

CSV_ESTUDIOS = textwrap.dedent("""\
    StudyInstanceUID,Report,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
    con-oro,ACL is torn.,1,0,1,0,0,0,0,1,0,0,0,0
    sin-oro,Normal knee.,,,,,,,,,,,,
    parcial,Something.,1,0,,,,,,,,,,
    """)

CSV_SERIES = (
    "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,"
    "Anatomical_Plane\n"
    "a,a1,1,1,Sagittal\n"
    "a,a2,1,1,Coronal\n"
    "b,b1,0,1,Sagittal\n"
)


@pytest.fixture
def csv_estudios(tmp_path: pathlib.Path) -> pathlib.Path:
    ruta = tmp_path / "train.csv"
    ruta.write_text(CSV_ESTUDIOS, encoding="utf-8")
    return ruta


@pytest.fixture
def csv_series(tmp_path: pathlib.Path) -> pathlib.Path:
    ruta = tmp_path / "train_series.csv"
    ruta.write_text(CSV_SERIES, encoding="utf-8")
    return ruta


def test_hay_doce_etiquetas_y_no_se_reordenan():
    # El orden ES la numeracion de clases en los pasos 2 a 5: reordenarlo
    # cambiaria el significado de cada indice del tensor de salida.
    assert len(LABELS) == 12
    assert LABELS[0] == "ACL"
    assert LABELS[-1] == "Fracture"


def test_carga_las_etiquetas_presentes_y_omite_las_vacias(csv_estudios):
    estudios = {e.uid: e for e in load_studies(csv_estudios)}
    assert len(estudios) == 3
    assert len(estudios["con-oro"].labels) == 12
    assert estudios["sin-oro"].labels == {}
    assert len(estudios["parcial"].labels) == 2


def test_solo_es_dorado_el_que_tiene_las_doce(csv_estudios):
    # 'parcial' tiene dos etiquetas: no alcanza. El criterio es TODAS, porque un
    # estudio a medio anotar no sirve como validacion limpia.
    oro = gold_studies(load_studies(csv_estudios))
    assert [e.uid for e in oro] == ["con-oro"]


def test_positives_devuelve_solo_los_unos(csv_estudios):
    oro = gold_studies(load_studies(csv_estudios))[0]
    assert oro.positives == frozenset({"ACL", "Medial Meniscus", "Effusion"})


def test_sagital_fluid_sensitive_exige_las_tres_condiciones(csv_series):
    series = {s.series_uid: s for s in load_series(csv_series)}
    assert series["a1"].is_sagittal_fluid_sensitive is True
    assert series["a2"].is_sagittal_fluid_sensitive is False   # coronal
    assert series["b1"].is_sagittal_fluid_sensitive is False   # no fluid-sens.


def test_study_y_series_son_inmutables():
    e = Study(uid="x", report="r", labels={})
    s = Series(study_uid="x", series_uid="y", fluid_sensitive=True,
               fat_suppression=True, plane="Sagittal")
    with pytest.raises(Exception):
        e.uid = "otro"          # type: ignore[misc]
    with pytest.raises(Exception):
        s.plane = "Coronal"     # type: ignore[misc]


def test_las_dos_copias_coinciden():
    """``pregunta1.estudios`` y ``rsna.selection`` sobre el CSV real.

    Es el test que justifica tener las dos. Si alguien arregla un caso borde en
    una y no en la otra, esto falla; sin él, el proyecto podría reportar dos
    conteos distintos de estudios dorados según qué módulo lo contó.
    """
    from pregunta1 import rutas
    if not rutas.TRAIN_CSV.is_file():
        pytest.skip("no estan los datos de la competencia")

    import rsna.selection as original
    import pregunta1.estudios as copia

    assert tuple(original.LABELS) == tuple(copia.LABELS)

    a = original.load_studies(rutas.TRAIN_CSV)
    b = copia.load_studies(rutas.TRAIN_CSV)
    assert len(a) == len(b)
    assert [x.uid for x in a] == [x.uid for x in b]
    assert [x.labels for x in a] == [x.labels for x in b]
    assert ([x.uid for x in original.gold_studies(a)]
            == [x.uid for x in copia.gold_studies(b)])

    if rutas.SERIES_CSV.is_file():
        sa = original.load_series(rutas.SERIES_CSV)
        sb = copia.load_series(rutas.SERIES_CSV)
        assert len(sa) == len(sb)
        assert ([x.is_sagittal_fluid_sensitive for x in sa]
                == [x.is_sagittal_fluid_sensitive for x in sb])
