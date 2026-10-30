# Paso 1 — Pipeline de extracción de etiquetas débiles

Resuelve el **entregable #1** del examen y produce los datos del **entregable #7**.

> **Pipeline de extracción de etiquetas débiles.** Diseñar e implementar un
> método (basado en reglas/palabras clave multilingües, o en *embeddings*
> multilingües con clasificación de texto de bajo costo) que extraiga las doce
> etiquetas objetivo a partir de los reportes en texto libre. Validar este
> pipeline comparando sus etiquetas contra los estudios con etiqueta dorada, y
> reportar una tasa de acuerdo/ruido estimada.

## Por qué este paso bloquea a todos los demás

Solo 58 de los 4,407 estudios tienen etiqueta verificada por radiólogo. Los otros
4,349 solo traen el reporte. Sin extraer etiquetas de ese texto no hay con qué
entrenar ningún modelo, ni con qué estratificar el subconjunto del paso 2.

## Clases y módulos de este paso

Todo el código de este paso vive en esta carpeta. Se importa como
`pregunta1.paso1_etiquetas_debiles.<modulo>` desde la raíz del repositorio.

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`oraciones.py`](oraciones.py) | `dividir_en_oraciones(texto)` | Parte el reporte en oraciones sin romper medidas (`8x12mm.`), decimales (`1.2x1.5cm.`) ni abreviaturas (`approx.`) |
| [`negacion.py`](negacion.py) | `EstadoAsercion`<br>`clasificar_asercion(oracion, inicio, fin)` | Clasifica una mención como `AFIRMADO`, `NEGADO` o `INCIERTO`, estilo NegEx: ventana de 6 palabras, disparadores previos (`no`, `without`, `sin`) y posteriores (`is intact`, `not torn`, `conservado`) |
| [`terminos.py`](terminos.py) | `Termino`<br>`TERMINOS`<br>`LABELS` | Diccionario de las 12 etiquetas en inglés y español. `Termino` separa **estructura** de **patología** |
| [`extractor.py`](extractor.py) | `ExtractorReglas`<br>`Extraccion`<br>`Evidencia` | Recorre oración por oración y arma las 12 etiquetas binarias con la evidencia que las sustenta |
| [`evaluacion.py`](evaluacion.py) | `Confusion`<br>`Reporte`<br>`evaluar(...)` | Matriz de confusión, acuerdo, tasa de ruido, precisión/recall/F1 |
| [`evaluar_extractor.py`](evaluar_extractor.py) | `main()` | Script ejecutable: mide el extractor contra los 58 dorados |
| [`baseline_palabras.py`](baseline_palabras.py) | `extraer_sin_negacion(reporte)`<br>`medir(oro, textos)` | Script ejecutable: el **baseline sin negación**, para medir cuánto aporta manejarla. Escribe `baseline_palabras.json` |
| [`generar_etiquetas.py`](generar_etiquetas.py) | `main()` | Script ejecutable: corre el extractor sobre los 4,407 reportes y escribe `etiquetas_debiles.csv` |
| [`traduccion.py`](traduccion.py) | `TraductorNLLB`<br>`Traduccion` | Traduce reportes al inglés con NLLB-200-distilled-600M, oración por oración |
| [`traducir_reportes.py`](traducir_reportes.py) | `main()` | Script ejecutable: traduce los 2,671 reportes no-ingleses y cachea en `reportes_traducidos.csv` |
| [`comparar_traduccion.py`](comparar_traduccion.py) | `main()` | Script ejecutable: mide el antes/después de traducir, por etiqueta y por idioma |
| [`comparar_estrategias.py`](comparar_estrategias.py) | `ESTRATEGIAS`<br>`textos_de(...)`<br>`main()` | Script ejecutable: las **cuatro** estrategias de traducción con el mismo código y el mismo conjunto |
| [`bootstrap_estrategias.py`](bootstrap_estrategias.py) | `main()` | Script ejecutable: bootstrap **pareado sobre estudios** del F1 macro, para saber si la mejora sobrevive a n=58 |
| [`normalidad.py`](normalidad.py) | `Veredicto`<br>`afirma_normalidad(...)`<br>`clasificar(...)`<br>`es_sano_verificado(...)` | Distingue una rodilla **sana** de un informe que el extractor no supo leer. Sin esto, los dos casos se ven igual en el CSV: todo en cero |
| [`diagnosticar_vacios.py`](diagnosticar_vacios.py) | `cargar_textos(...)`<br>`clasificar(...)`<br>`main()` | Script ejecutable: por qué el extractor deja 807 estudios sin ninguna etiqueta. Escribe `diagnostico_vacios.json` |

Tests en `tests/test_etiquetas_oraciones.py` y `tests/test_etiquetas_negacion.py`.

## Comandos

Todos se corren **desde la raíz del repositorio** (`D:\Backend
edesneuronales`),
con el venv del proyecto.

### Medir el extractor contra los 58 dorados

```bash
.venv/Scripts/python.exe pregunta1/paso1_etiquetas_debiles/evaluar_extractor.py
```

Imprime la matriz de confusión por etiqueta, el acuerdo macro, la tasa de ruido
estimada y el desglose por idioma.

### Traducir los reportes no-ingleses (una sola vez, ~45 min en GPU)

```bash
.venv/Scripts/python.exe pregunta1/paso1_etiquetas_debiles/traducir_reportes.py
```

Escribe `reportes_traducidos.csv`. La primera corrida descarga el modelo (2.4 GB).

### Medir qué aportó la traducción

```bash
.venv/Scripts/python.exe pregunta1/paso1_etiquetas_debiles/comparar_traduccion.py
```

### Generar las etiquetas débiles (el producto del paso)

```bash
.venv/Scripts/python.exe pregunta1/paso1_etiquetas_debiles/generar_etiquetas.py
```

Escribe `etiquetas_debiles.csv` con una fila por estudio (4,407) y columnas:

| Columna | Qué es |
|---|---|
| `StudyInstanceUID` | Identificador del estudio |
| `idioma` | Idioma detectado del reporte |
| `n_oraciones` | Oraciones en que se partió el reporte |
| `n_evidencias` | Menciones que el extractor encontró — **señal de confianza** para la ablación del entregable #4 |
| `es_dorado` | 1 para los 58 con etiqueta de radiólogo |
| Las 12 etiquetas | 0/1 extraídos del texto |

Tarda ~13 segundos.

### Correr los tests de este paso

```bash
.venv/Scripts/python.exe -m pytest tests/test_etiquetas_oraciones.py tests/test_etiquetas_negacion.py -q
```

### Usar las clases desde Python

```python
from pregunta1.paso1_etiquetas_debiles import ExtractorReglas

extractor = ExtractorReglas()
resultado = extractor.extraer("The ACL is intact. Horizontal tear of the lateral meniscus.")

print(sorted(resultado.positivos()))          # ['Lateral Meniscus']
print(resultado.etiquetas["ACL"])             # 0  -> la negacion funciono

for evidencia in resultado.evidencias:        # por que decidio cada etiqueta
    print(evidencia.etiqueta, evidencia.estado.value, evidencia.oracion)
```

`ExtractorReglas(incierto_es_positivo=True)` cuenta las menciones dudosas
("possible tear") como positivas; por defecto no lo hace.

---

## Lo que se hizo

### 1. Caracterización del problema (medida, no supuesta)

Se detectó el idioma de los 4,407 reportes con `langdetect`:

```
inglés 39.4%  ·  español 15.5%  ·  turco 12.4%  ·  croata 9.2%  ·  griego 7.3%
alemán 5.9%   ·  búlgaro 5.0%   ·  neerlandés 3.5%  ·  francés 1.8%
```

Nueve idiomas, **dos sin alfabeto latino** (griego, búlgaro). Los 58 dorados
están repartidos en ocho: 28 inglés, 10 español, 6 turco, 4 croata, 3 griego,
3 búlgaro, 2 neerlandés, 2 alemán. No se puede validar solo en inglés.

### 2. Se midió por qué un buscador de palabras clave no sirve

Se corrió un baseline que solo busca el término, sin manejar negación. Lee **el
mismo texto traducido** que la producción y usa **los mismos términos**, así que
la única diferencia es la negación:

```bash
.venv/Scripts/python.exe pregunta1/paso1_etiquetas_debiles/baseline_palabras.py
```

| etiqueta | FP baseline | FP extractor | acuerdo baseline | acuerdo extractor | recall |
|---|---:|---:|---:|---:|---:|
| Medial Meniscus | 16 | **12** | 62.1% | **69.0%** | 0.77 (igual) |
| Effusion | 16 | **11** | 48.3% | **56.9%** | 0.60 (igual) |
| Lateral Meniscus | 15 | **14** | 62.1% | **63.8%** | 0.70 (igual) |
| Contusion | 13 | **10** | 70.7% | **75.9%** | 0.79 (igual) |
| Fracture | 11 | **5** | 74.1% | **84.5%** | 0.78 (igual) |
| Baker's | 9 | **4** | 77.6% | **86.2%** | 0.67 (igual) |
| ACL | 8 | **7** | 72.4% | **74.1%** | 0.67 (igual) |
| PF OA | 8 | **8** | 69.0% | **69.0%** | 0.52 (igual) |
| MCL | 7 | **6** | 86.2% | **87.9%** | 0.89 (igual) |
| Synovitis | 4 | **4** | 63.8% | **63.8%** | 0.37 (igual) |
| Medial OA | 3 | **1** | 72.4% | **75.9%** | 0.13 (igual) |
| Lateral OA | 2 | **1** | 82.8% | **84.5%** | 0.27 (igual) |
| **total / macro** | **112** | **83** | **70.1%** | **74.3%** | |

**Manejar la negación elimina 29 de los 112 falsos positivos (26%) sin perder un
solo verdadero positivo: el recall es idéntico en las doce etiquetas.** El
acuerdo macro sube de 70.1% a 74.3%. No hay compromiso que discutir acá, que es
un resultado más fuerte —y más raro— que una mejora con costo.

La causa está en cómo se redacta un informe radiológico: **enumera lo que NO
encontró**.

```
"ACL is intact."              → el matcher ve "ACL"   → marca 1   REAL: 0
"The MCL is intact."          → ve "MCL"              → marca 1   REAL: 0
"Medial meniscus is not torn" → ve "medial meniscus"  → marca 1   REAL: 0
"No fracture is seen."        → ve "fracture"         → marca 1   REAL: 0
```

**Corrección.** Una versión anterior de este README publicaba otra tabla acá
—Medial Meniscus 18 FP / 44.8%, Effusion 16 / 46.6%, ACL 18 / 51.7%, MCL 20 /
60.3%— y concluía *"peor que una moneda en varios casos"*. Tres de esas cuatro
filas no se reproducen (ACL tiene 8 FP, no 18; MCL 7, no 20), **no había ningún
script en el repositorio que las produjera**, y solo **una** etiqueta queda por
debajo del 50% de acuerdo: Effusion, con 48.3%. `baseline_palabras.py` existe
para que esto se pueda volver a medir en vez de tener que creerlo.

### 3. Se construyó el extractor por reglas

Cinco módulos en esta misma carpeta (ver la tabla de arriba para el detalle):

| Módulo | Responsabilidad |
|---|---|
| `oraciones.py` | Parte el reporte en oraciones sin romper medidas (`8x12mm.`), decimales (`1.2x1.5cm.`) ni abreviaturas (`approx.`) |
| `negacion.py` | Clasifica cada mención como **afirmada**, **negada** o **incierta**, estilo NegEx: ventana de 6 palabras, disparadores previos (`no`, `without`, `sin`) y posteriores (`is intact`, `not torn`, `conservado`) |
| `terminos.py` | Diccionario de las 12 etiquetas en inglés y español |
| `extractor.py` | Recorre oración por oración y arma las 12 etiquetas binarias con su evidencia |
| `evaluacion.py` | Matriz de confusión, acuerdo, tasa de ruido, precisión/recall/F1 |

**La decisión de diseño que más impacto tuvo:** separar *estructura* de
*patología*. Siete de las doce etiquetas exigen que ambas aparezcan en la misma
oración, porque nombrar la estructura no prueba nada —todo informe menciona el
ACL, casi siempre para decir que está sano—:

```python
"ACL": Termino(
    estructura=_re(r"\bACL\b|anterior cruciate|ligamento cruzado anterior"),
    patologia=_re(r"\btear\b|\btorn\b|rotura|desgarro|..."),   # ambas, misma oración
)
"Effusion": Termino(
    estructura=_re(r"\beffusion\b|derrame"),                   # el término ya es el hallazgo
)
```

Las cinco etiquetas restantes (Effusion, Synovitis, Baker's, Contusion, Fracture)
son hallazgos directos: el término mismo es la patología.

### 4. Se validó contra los 58 dorados

```
etiqueta            TP  FP  FN  TN  acuerdo   prec    rec     F1
------------------------------------------------------------------
ACL                 13   5  11  29   72.4%   0.72   0.54   0.62
MCL                  7   4   2  45   89.7%   0.64   0.78   0.70
Medial Meniscus     12  11  14  21   56.9%   0.52   0.46   0.49
Lateral Meniscus    13   7  10  28   70.7%   0.65   0.57   0.60
Medial OA            2   1  13  42   75.9%   0.67   0.13   0.22
Lateral OA           3   1   8  46   84.5%   0.75   0.27   0.40
PF OA               11   8  10  29   69.0%   0.58   0.52   0.55
Effusion            20  11  15  12   55.2%   0.65   0.57   0.61
Synovitis           10   4  17  27   63.8%   0.71   0.37   0.49
Baker's              4   3   8  43   81.0%   0.57   0.33   0.42
Contusion           10   9   9  30   69.0%   0.53   0.53   0.53
Fracture             7   2  11  38   77.6%   0.78   0.39   0.52
------------------------------------------------------------------
MACRO                                72.1%                 0.51
```

> **TASA DE RUIDO ESTIMADA (macro): 27.9%** ← el número que pide el entregable #1

Mejoras frente al baseline ingenuo: **MCL +29 puntos**, **ACL +21**,
**Fracture +10**, **Baker's +7**.

### 5. Se midió el sesgo por idioma (entregable #7)

```
idioma      n   acuerdo  F1 macro
---------------------------------
en         28    78.9%      0.66
es         10    80.8%      0.48
tr          6    62.5%      0.04
hr          4    60.4%      0.00
el          3    47.2%      0.00
bg          3    66.7%      0.00
nl          2    41.7%      0.06
de          2    62.5%      0.08
```

**El hallazgo central del paso.** El diccionario cubre inglés y español; en los
otros siete idiomas el F1 es prácticamente **cero**. El acuerdo no cae a cero
solo porque acertar "negativo" en una etiqueta rara ya da acuerdo alto sin
detectar nada — razón por la cual el acuerdo solo es una métrica engañosa y hay
que mirar el F1 al lado.

Esto es exactamente el riesgo de sesgo lingüístico que pide discutir el
entregable #7, medido en vez de especulado: el pipeline produciría etiquetas
útiles para el 55% de los estudios y ruido puro para el 45% restante.

---

### 6. Se generaron las etiquetas para los 4,407 estudios

Prevalencia extraída frente a la real en los 58 dorados:

| etiqueta | extraída (n=4 407) | real (n=58) | extraída / real |
|---|---:|---:|---:|
| Medial Meniscus | 45.6% | 44.8% | **1.02** |
| Effusion | 33.5% | 60.3% | 0.56 |
| Lateral Meniscus | 26.3% | 39.7% | 0.66 |
| PF OA | 24.7% | 36.2% | 0.68 |
| Baker's | 17.3% | 20.7% | 0.84 |
| ACL | 16.8% | 41.4% | 0.41 |
| Contusion | 14.0% | 32.8% | 0.43 |
| MCL | 10.8% | 15.5% | 0.69 |
| Synovitis | 10.3% | 46.6% | **0.22** |
| Fracture | 7.9% | 31.0% | 0.25 |
| Medial OA | 7.5% | 25.9% | 0.29 |
| Lateral OA | 4.6% | 19.0% | 0.24 |

El extractor **subdetecta, pero no de forma uniforme**, y la diferencia importa
para la ablación del entregable #4: el ratio va de **1.02** (Medial Meniscus, que
no subdetecta nada) a **0.22** (Synovitis, que pierde cuatro de cada cinco).

Las tres peores —Synovitis 0.22, Lateral OA 0.24, Fracture 0.25— no fallan por la
misma razón:

- **Synovitis** es la única donde el recall medido también se hunde (**0.37**): el
  extractor de verdad no la encuentra.
- **Fracture** tiene recall **0.78**, de los mejores. Extrae poco porque la
  fractura es poco prevalente en los 4 407, no porque falle en detectarla.

Esa distinción es la que separa "la etiqueta es ruidosa" de "la etiqueta es rara",
y son dos problemas distintos con dos soluciones distintas.

**Corrección.** Una versión anterior de este README daba otra columna de
prevalencia extraída —Effusion 30.7%, Medial Meniscus 24.1%, PF OA 21.0%,
Fracture 4.1%— y **ninguna de esas once cifras se reproduce** contra
`etiquetas_debiles.csv`. Cambiaba además el orden: la etiqueta más extraída es
**Medial Meniscus (45.6%)**, no Effusion. La conclusión que sacaba —*"la
prevalencia extraída es entre la mitad y un cuarto de la real en todas las
etiquetas"*— es falsa en los dos extremos: Medial Meniscus está en 1.02 y
Synovitis en 0.22. Los recalls que citaba (ACL 0.54, Fracture 0.39) tampoco:
`evaluar_extractor.py` mide **ACL 0.67** y **Fracture 0.78**; solo Synovitis
(0.37) estaba bien.

### 7. Se tradujeron los reportes y se midió el impacto

Se tradujeron los 2,671 reportes no-ingleses con **NLLB-200-distilled-600M**,
oración por oración (los reportes llegan a 4,743 caracteres, más de lo que el
modelo procesa de una vez). ~45 min en la GPU, cacheado en
`reportes_traducidos.csv` para no repetirlo.

**Impacto sobre los 58 dorados, por idioma:**

| idioma | n | F1 antes | F1 después | delta |
|---|---|---|---|---|
| inglés | 28 | 0.66 | 0.66 | +0.00 |
| **español** | 10 | **0.48** | **0.29** | **−0.19** |
| turco | 6 | 0.04 | 0.34 | +0.30 |
| croata | 4 | 0.00 | 0.21 | +0.21 |
| griego | 3 | 0.00 | **0.11** | **+0.11** |
| búlgaro | 3 | 0.00 | 0.32 | +0.32 |
| neerlandés | 2 | 0.06 | 0.44 | +0.39 |
| alemán | 2 | 0.08 | **0.19** | **+0.11** |

**El español empeoró al traducirlo.** Era el control que se diseñó para esto:
el español funcionaba nativo al 0.48, así que si la traducción no rompiera nada
debería quedar parecido. Cayó 19 puntos — NLLB degrada terminología médica. El
inglés quedó idéntico, lo que confirma que el pipeline no altera lo que no toca.

Ejemplos de degradación observados en la muestra:

```
turco:  "Left eye MRG"              <- deberia decir knee
turco:  "cross-linked ligaments"    <- cruciate
turco:  "minimal urinary growth"    <- probablemente effusion
griego: "MRI GERMAN TECHNOLOGY"     <- basura
griego: "the artery cartilage"      <- articular
```

### 8. Se eligió la estrategia midiendo las cuatro opciones

```bash
.venv/Scripts/python.exe pregunta1/paso1_etiquetas_debiles/comparar_estrategias.py
```

| estrategia | acuerdo | F1 macro | ruido |
|---|---:|---:|---:|
| sin traducir | 72.1% | 0.512 | 27.9% |
| traducir todo | 74.0% | 0.556 | 26.0% |
| traducir salvo inglés | 74.0% | 0.556 | 26.0% |
| **traducir salvo inglés y español** | **74.3%** | **0.589** | **25.7%** |

Gana el híbrido: traducir solo los siete idiomas que el extractor no lee
nativamente. Es lo que quedó como comportamiento por defecto de
`generar_etiquetas.py` (constante `IDIOMAS_NATIVOS`).

**Las filas 2 y 3 dan idéntico, y eso es un resultado, no un empate de
redondeo.** Hay traducción cacheada para los 4 407 estudios, inglés incluido, así
que "traducir todo" **sí** pasa los 28 dorados en inglés por NLLB. Que el número
no se mueva significa que traducir inglés a inglés no les cambia nada medible. Lo
que mueve la aguja es exclusivamente **no** traducir el español.

**Verificado con bootstrap pareado sobre estudios**, porque con n=58 una
diferencia de 8 puntos de F1 podría ser ruido:

```bash
.venv/Scripts/python.exe pregunta1/paso1_etiquetas_debiles/bootstrap_estrategias.py
```

```
F1 sin traducir : 0.512   IC95% [0.416, 0.578]
F1 hibrido      : 0.589   IC95% [0.518, 0.636]
delta pareado   : +0.076  IC95% [+0.039, +0.127]
P(hibrido mejor): 100.0%  sobre 2,000 remuestreos
```

El intervalo de la **diferencia** no cruza cero: la mejora es real. Los dos
intervalos sueltos, en cambio, se solapan ([0.416, 0.578] contra [0.518, 0.636]),
que es exactamente la razón de remuestrear pareado y no compararlos a ojo.

**Corrección.** Una versión anterior de este README daba las cuatro filas
corridas —le atribuía al híbrido el 74.0% que le corresponde a "traducir todo", y
ponía 73.7% / 0.551 / 26.3% en las dos filas del medio, que miden 74.0% / 0.556 /
26.0%—. El bootstrap que citaba (F1 0.584, delta +0.072, IC [+0.035, +0.122])
**no tenía ningún script en el repositorio**; medido ahora da 0.589, +0.076 e IC
[+0.039, +0.127]. La conclusión no cambia; la diferencia es que ahora se puede
volver a correr.

### 9. El 46.8% de estudios vacíos bajó al 18.3%

| idioma | n | vacíos antes | vacíos ahora |
|---|---:|---:|---:|
| búlgaro | 220 | 99.5% | **10.9%** |
| croata | 406 | 93.1% | **9.4%** |
| neerlandés | 153 | 94.8% | **17.6%** |
| turco | 546 | 87.5% | **25.1%** |
| francés | 81 | 71.6% | **9.9%** |
| alemán | 262 | 63.7% | **26.7%** |
| **griego** | 321 | 90.3% | **54.2%** |
| inglés | 1 736 | 11.5% | 11.5% |
| español | 682 | 18.9% | 18.9% |

De 2 064 estudios sin etiqueta a **807** (18.3% de 4 407). El griego sigue siendo
el peor caso con diferencia —**54.2%**, el doble del segundo peor— consistente con
la mala calidad de su traducción.

**Corrección.** Una versión anterior de este README decía *"bajó al 20.1%"*, con
**886** estudios vacíos, griego **68.2%**, alemán 36.3% y francés 21.0%. Medido
sobre `etiquetas_debiles.csv`: **807** vacíos (**18.3%**), griego **54.2%**,
alemán **26.7%**, francés **9.9%**. Las otras cinco filas sí se reproducen. El
`886`/`20.1%` aparecía además repetido más abajo y en el README raíz de la
pregunta; está corregido en los tres lugares.

## Hallazgos que afectan al resto del proyecto

### El problema de los estudios sin etiqueta (estado original, antes de traducir)

2 064 de 4 407 (46.8%); hoy son **807 (18.3%)** tras aplicar el híbrido. Y el
desglose por idioma muestra que no son estudios sanos — son estudios donde el
extractor está **ciego**:

| idioma | n | sin ninguna etiqueta | evidencias por estudio |
|---|---|---|---|
| inglés | 1,736 | 11.5% | 6.1 |
| español | 682 | 18.9% | 4.6 |
| alemán | 262 | 63.7% | 0.4 |
| francés | 81 | 71.6% | 0.8 |
| turco | 546 | **87.5%** | 0.1 |
| griego | 321 | **90.3%** | 0.1 |
| croata | 406 | **93.1%** | 0.1 |
| neerlandés | 153 | **94.8%** | 0.1 |
| búlgaro | 220 | **99.5%** | 0.0 |

En búlgaro el extractor encuentra **cero** evidencias en el 99.5% de los casos.

**Consecuencia directa para el paso 2:** entrenar con esas filas le enseñaría al
modelo que 2,064 estudios con patología real son todo-negativos — peor que
excluirlos. Hay dos salidas, y la elección es parte del entregable #4:

1. **Excluir los estudios sin evidencia** → quedan ~2,343, casi todos en inglés y
   español. Destraba el paso 2 ya, pero mete sesgo lingüístico en el propio
   conjunto de entrenamiento.
2. **Traducir al inglés primero** → recupera los otros siete idiomas.

La opción 1 es, en sí misma, el caso de estudio que pide el entregable #7:
un pipeline de etiquetas que falla por idioma termina produciendo un modelo
entrenado casi solo con datos de centros angloparlantes e hispanohablantes.


### Las etiquetas doradas no son una transcripción literal del reporte

En un estudio dorado, el texto dice explícitamente:

```
"Moderate joint effusion ... are observed."    →  Effusion = 0  en la etiqueta dorada
"Osteochondral fracture at lateral patellar"   →  Fracture = 0  en la etiqueta dorada
```

Hay **criterio clínico** de por medio: una fractura osteocondral en la faceta
rotuliana se cuenta como osteoartrosis patelofemoral, no como fractura. Ningún
pipeline de texto, por bueno que sea, puede llegar al 100%: existe un **techo
irreducible**. Debe quedar documentado en la discusión crítica del informe.

### Medial OA y Lateral OA tienen recall muy bajo (0.13 y 0.27)

El extractor casi no los detecta. La osteoartrosis por compartimento se describe
con rodeos ("condropatía focal grado 4 del aspecto inferior de la vertiente
medial de la tróclea femoral") donde el compartimento y el daño cartilaginoso
están a varias oraciones de distancia, o implícitos en la anatomía nombrada.
Es el caso más difícil de los doce.

### Una regla que se probó y se descartó

Partir también por secuencias de 3+ espacios. Medido contra los dorados a
umbrales de 2, 3, 4, 6 y 8 espacios: el acuerdo macro se quedó en 72.1% y el F1
en 0.512 **en los cinco casos**, porque esas secuencias siempre acompañan a un
punto que ya cierra la oración. Se eliminó en lugar de dejarla con un umbral
elegido a dedo.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | ~~Traducir los reportes al inglés~~ | ✅ **hecho** — híbrido, F1 0.512 → 0.589 |
| 2 | Vía B: embeddings multilingües + clasificador | El examen autoriza las dos vías; compararlas alimenta la ablación del entregable #4 |
| 3 | ~~No calibrar contra los mismos 58 que se reportan~~ | ✅ **hecho** — `particionar_dorados` parte los 58 en mitad de desarrollo y mitad reservada; ver abajo |
| 4 | Mejorar Medial OA / Lateral OA | Recall de 0.13 y 0.27; son las dos peores |
| 5 | ~~Generar `etiquetas_debiles.csv`~~ | ✅ **hecho** — 4 407 filas |
| 6 | ~~Decidir qué hacer con los estudios sin evidencia~~ | ✅ **hecho** — bajaron de 2 064 a 807 traduciendo; el paso 2 los usa como señal de confianza (`n_evidencias`) |

**Sobre el punto 3, y el número que importa.** Ajustar el extractor midiendo
contra los 58 y después reportar el acuerdo sobre esos mismos 58 lo ajusta a
ellos. `evaluar_extractor.py --parte {desarrollo,reservados}` separa las dos
mitades, y la diferencia se ve:

| mitad | n | acuerdo macro | F1 macro |
|---|---:|---:|---:|
| desarrollo (mirada muchas veces) | 29 | 75.9% | **0.60** |
| **reservada (se toca una vez)** | 29 | 72.7% | **0.56** |
| todos | 58 | 74.3% | 0.59 |

**El 0.59 que se reporta sobre los 58 está ~0.03 por encima del 0.56 de la mitad
que nunca se usó para ajustar.** Es poco, pero es en la dirección esperada, y es
la razón de reportar las tres filas en vez de solo la última. Lo que falta de
verdad para cerrar el punto es rotar 5 particiones en vez de partir en dos; con
29 estudios por mitad, cada número tiene un intervalo ancho.

Quedan como pendientes reales la vía B con embeddings (2), Medial OA / Lateral OA
(4) y el griego, que sigue al **54.2%** de estudios vacíos.
