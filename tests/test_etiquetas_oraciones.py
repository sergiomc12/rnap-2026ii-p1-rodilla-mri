from pregunta1.paso1_etiquetas_debiles.oraciones import dividir_en_oraciones


def test_separa_por_punto_final():
    assert dividir_en_oraciones("The ACL is intact. No fracture is seen.") == [
        "The ACL is intact.",
        "No fracture is seen.",
    ]


def test_no_parte_una_medida_con_punto():
    texto = "Tear about 8x12mm. is noted."

    assert dividir_en_oraciones(texto) == ["Tear about 8x12mm. is noted."]


def test_no_parte_un_decimal():
    assert len(dividir_en_oraciones("Defect 1.2x1.5cm at the trochlea.")) == 1


def test_no_parte_una_abreviatura_conocida():
    assert len(dividir_en_oraciones("Lesion approx. 4mm at the horn.")) == 1


def test_los_espacios_no_separan_por_si_solos():
    """Measured as worthless against the gold set; see the module docstring."""
    texto = "ACL is intact     Medial meniscus is torn"

    assert dividir_en_oraciones(texto) == ["ACL is intact Medial meniscus is torn"]


def test_colapsa_espacios_internos():
    assert dividir_en_oraciones("The   ACL  is intact.") == ["The ACL is intact."]


def test_descarta_fragmentos_triviales():
    assert dividir_en_oraciones("a. The ACL is intact.") == ["The ACL is intact."]


def test_texto_vacio_no_produce_oraciones():
    assert dividir_en_oraciones("   ") == []
