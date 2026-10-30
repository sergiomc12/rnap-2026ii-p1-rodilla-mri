# Paso 2 — Muestreo estratificado y preprocesamiento

Produce el caché de tensores que consumen los modelos del paso 3.

Resuelve la **"Reducción de Escala Recomendada"** del examen:

> Restringir el trabajo a uno o dos planos/secuencias de MRI · Trabajar con cortes
> 2D (o un pequeño volumen 2.5D) · Construir un subconjunto estratificado de
> estudios (sugerido: entre 300 y 600) **asegurando representación de casos
> positivos para cada uno de los doce hallazgos** · Reservar los ~58 estudios con
> etiqueta dorada como conjunto de validación limpio.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`lector_zip.py`](lector_zip.py) | `LectorZip`<br>`EntradaDicom` | Acceso aleatorio al ZIP de 570 GB sin descomprimirlo. Indexa 819,078 DICOM en 2.8 s |
| [`muestreo.py`](muestreo.py) | `MuestreadorEstratificado`<br>`EstudioEtiquetado`<br>`Muestra` | Elige los estudios garantizando positivos de las 12 etiquetas |
| [`geometria.py`](geometria.py) | `MetadatosCorte`<br>`recortar_milimetros()`<br>`normalizar_intensidad()`<br>`es_rodilla_izquierda()`<br>`ordenar_por_posicion()`<br>`banda_central()` | Lleva todos los estudios al mismo marco físico |
| [`dicom.py`](dicom.py) | `leer_dicom()`<br>`metadatos_de()`<br>`pixeles_de()` | Decodifica bytes DICOM tolerando tags ausentes y sintaxis comprimidas |
| [`preprocesamiento.py`](preprocesamiento.py) | `ConfigPreproceso`<br>`preprocesar_serie()`<br>`indices_de_grupos()`<br>`redimensionar()` | Serie DICOM → tensor 2.5D `(8, 3, 224, 224)` |
| [`seleccionar_estudios.py`](seleccionar_estudios.py) | `main()` | Script: elige la muestra y escribe `estudios_seleccionados.csv` |
| [`generar_cache.py`](generar_cache.py) | `main()` | Script: preprocesa y cachea los `.npy` |

Tests en `tests/test_paso2_geometria.py`, `test_paso2_preprocesamiento.py` y
`test_paso2_muestreo.py` (42 tests).

## Comandos

Desde la raíz del repositorio.

### 1. Elegir los estudios

```bash
.venv/Scripts/python.exe pregunta1/paso2_preprocesamiento/seleccionar_estudios.py
.venv/Scripts/python.exe pregunta1/paso2_preprocesamiento/seleccionar_estudios.py --objetivo 400 --cuota 30
```

Escribe `estudios_seleccionados.csv`: una fila por estudio con su serie sagital,
idioma, si es dorado, cantidad de cortes y las 12 etiquetas.

### 2. Generar el caché de tensores

```bash
.venv/Scripts/python.exe pregunta1/paso2_preprocesamiento/generar_cache.py
.venv/Scripts/python.exe pregunta1/paso2_preprocesamiento/generar_cache.py --limite 20
.venv/Scripts/python.exe pregunta1/paso2_preprocesamiento/generar_cache.py --rehacer
```

Es **reanudable**: un `.npy` que ya existe se salta, salvo con `--rehacer`.

### 3. Correr los tests del paso

```bash
.venv/Scripts/python.exe -m pytest tests/test_paso2_geometria.py tests/test_paso2_preprocesamiento.py tests/test_paso2_muestreo.py -q
```

### Usar las clases desde Python

```python
from pregunta1.paso2_preprocesamiento.lector_zip import LectorZip
from pregunta1.paso2_preprocesamiento.dicom import leer_dicom, metadatos_de, pixeles_de
from pregunta1.paso2_preprocesamiento.preprocesamiento import ConfigPreproceso, preprocesar_serie

with LectorZip() as lector:
    cortes = lector.cortes_de("<SeriesInstanceUID>")
    conjuntos = [leer_dicom(lector.leer(c)) for c in cortes]
    tensor = preprocesar_serie(
        [pixeles_de(d) for d in conjuntos],
        [metadatos_de(d) for d in conjuntos],
        ConfigPreproceso(),
    )
print(tensor.shape)   # (8, 3, 224, 224)
```

---

## Lo que se hizo

### 1. Acceso al ZIP sin descomprimirlo

`LectorZip` lee cualquier entrada por nombre desde el directorio central del
archivo. **Indexa los 819,078 DICOM de entrenamiento en 2.8 segundos** y después
cada lectura es un acceso directo.

Los 570 GB nunca se descomprimen a disco. Como referencia de por qué importa:
bajar el mismo subconjunto por la API de Kaggle tomó 6 h 25 min para 1.39 GB y
falló en el 63% de los archivos por rate limiting.

### 2. Muestreo estratificado en tres pasadas

Un muestreo aleatorio de 600 no sirve: **Lateral OA tiene 203 positivos entre los
4 407**, así que caerían unos 28 — muy pocos para aprender y menos aún para
medir.

1. **Entran todos los dorados.** Son la validación limpia; van sí o sí, sin
   importar qué digan las etiquetas débiles sobre ellos.
2. **Cuota por etiqueta, de la más rara a la más común.** El orden importa: un
   estudio positivo en Fracture suele ser positivo también en algo frecuente, así
   que las comunes se llenan solas mientras que las raras nunca lo harían.
3. **El resto al azar**, para que la muestra represente la distribución real y no
   solo patología.

**Se excluyen los estudios sin ninguna evidencia** (`n_evidencias == 0`). No son
rodillas sanas: son reportes que el extractor no pudo leer. Entrenar con ellos le
enseñaría al modelo que estudios con patología real son todo-negativos.

Resultado:

```
MUESTRA DE 600 ESTUDIOS
  dorados           : 58
  por cuota         : 96
  al azar           : 446
  sanos verificados : 1
  descartados ilegibles: 195
```

**Hay dos conteos y no son el mismo, así que van los dos.** El muestreador elige
600 estudios; después se le busca a cada uno su serie **sagital
fluid-sensitive**, y **27 no tienen ninguna**. Lo que llega al caché y al
entrenamiento son **573**, que es lo que hay en `estudios_seleccionados.csv`:

| etiqueta | pos. en la muestra (600) | pos. en el CSV (573) | prevalencia del CSV |
|---|---:|---:|---:|
| Lateral OA | 68 | **65** | 11.3% |
| Medial OA | 83 | **80** | 14.0% |
| Fracture | 83 | **81** | 14.1% |
| MCL | 90 | **88** | 15.4% |
| Synovitis | 92 | **91** | 15.9% |
| Contusion | 122 | **117** | 20.4% |
| Baker's | 132 | **123** | 21.5% |
| ACL | 149 | **149** | 26.0% |
| PF OA | 194 | **183** | 31.9% |
| Lateral Meniscus | 213 | **208** | 36.3% |
| Effusion | 263 | **259** | 45.2% |
| Medial Meniscus | 337 | **331** | 57.8% |

**Ninguna etiqueta baja de 65 positivos**, por encima del mínimo que exige el
enunciado. La más rara de la muestra (Lateral OA) captura el **32%** de todos los
positivos de esa etiqueta que existen en el dataset.

**Y los dorados bajan de 58 a 56.** Dos de los 58 estudios con etiqueta de
radiólogo no tienen serie sagital fluid-sensitive, así que no entran al caché.
Por eso los AUC de los pasos 3 y 4 se reportan **sobre 56**, no 58, mientras que
el acuerdo del extractor del paso 1 —que solo necesita el texto del informe— se
reporta sobre los 58.

**Corrección.** Una versión anterior de este README publicaba una tercera tabla
—Lateral OA 73, Medial OA 84, Fracture 88, MCL 102, ... Medial Meniscus 304— que
**no coincide con ninguna de las dos**, junto con `descartados vacios: 604`,
un campo que el muestreador ya no imprime (hoy son `sanos verificados` y
`descartados ilegibles`). Describía una versión anterior del muestreador. También
decía que *"Fracture tiene 182 positivos entre 4,407"*: son **348**, y la etiqueta
más rara del dataset es **Lateral OA con 203**, que es la que justifica la cuota.

### 3. Normalización geométrica — lo que más impacto tiene

Medido sobre 764 cortes reales, las adquisiciones no coinciden en casi nada:

| | |
|---|---|
| Resoluciones | 10 distintas, de 320×320 a 1024×1024 |
| `PixelSpacing` | 0.146 a 0.562 mm/px — **factor 4** |
| Campo de visión físico | 130 a 240 mm (mediana 170) |
| Fabricantes | 8, con nombres sin normalizar |
| Campo magnético | 1.5T y 3T mezclados |
| Profundidad | 12 y 16 bits mezclados |
| `Laterality` | **ausente o vacío en el 54%** |

Redimensionar eso directo a 224×224 le entregaría a la red el mismo menisco a
escalas físicas muy distintas: tendría que aprender configuraciones de escáner
antes de poder aprender anatomía.

**`recortar_milimetros()` recorta 130 mm físicos usando `PixelSpacing`, y recién
después redimensiona.** Verificado sobre tres series reales muy distintas:

```
320×320 esp=0.500mm FOV=160mm  ->  (8, 3, 224, 224)
800×800 esp=0.225mm FOV=180mm  ->  (8, 3, 224, 224)
640×640 esp=0.375mm FOV=240mm  ->  (8, 3, 224, 224)
```

Una adquisición más chica que el recorte se devuelve intacta en vez de rellenarse:
rellenar inventaría anatomía que nunca se escaneó.

### 4. Normalización de intensidad por corte

La MRI **no tiene unidades absolutas** — el mismo tejido lee 636 en un escáner y
2781 en otro. `normalizar_intensidad()` escala cada corte a [0,1] con sus propios
percentiles 1 y 99; recortar en el 99 evita que un artefacto brillante aplaste
todo el resto del rango.

### 5. Lateralidad, con fallback honesto

El tag `Laterality` falta en el 54% de los cortes. `es_rodilla_izquierda()` lo
intenta en dos fuentes y **devuelve `None` si ninguna responde**, para que el
llamador deje la serie como está en vez de voltearla adivinando:

1. El tag `Laterality` (`L`/`R`).
2. El sistema de coordenadas del paciente: `+x` apunta a la izquierda, así que una
   rodilla izquierda se escanea en x positivo.

En los cortes sagitales, una rodilla corre medial→lateral y la otra al revés. Sin
invertir una de las dos, "el tercer corte" significaría anatomías distintas según
el estudio.

### 6. Salida 2.5D con grupos de 3 cortes

```
(grupos=8, canales=3, alto=224, ancho=224)   float32 en [0,1]   4.8 MB por estudio
```

**Tres canales y no cinco ni siete** porque los pesos de ImageNet esperan
exactamente tres: así la primera convolución preentrenada se carga sin tocar. Es
lo que mantiene comparables al modelo desde cero y al de transfer learning, que es
justo lo que exige la comparación rigurosa del entregable #3.

Los 8 grupos se reparten a lo largo de la banda central (20%–80% de los cortes,
descartando los extremos que son casi todo aire y diáfisis), y cada grupo son
cortes **contiguos** — eso es lo que lo hace 2.5D y no tres imágenes sin relación.

Una serie más corta que el grupo repite el último corte en vez de perder el
estudio.

### 7. Caché generado

```
procesados=576  saltados=0  fallidos=0  |  2.77 GB  |  1.7 min
```

**Cero fallos de decodificación** en los 576 estudios. El caché es reanudable: un
`.npy` existente se salta salvo con `--rehacer`.

---

## Hallazgos y decisiones

### La variante coronal, para la ablación de plano

`seleccionar_estudios.py` acepta `--plano {Sagittal,Coronal,Axial}`, y
`generar_cache.py` acepta `--seleccion` y `--cache`, de modo que construir el
conjunto coronal no necesitó código nuevo:

```bash
.venv/Scripts/python.exe pregunta1/paso2_preprocesamiento/seleccionar_estudios.py \
    --plano Coronal --salida pregunta1/paso2_preprocesamiento/estudios_coronal.csv

.venv/Scripts/python.exe pregunta1/paso2_preprocesamiento/generar_cache.py \
    --seleccion pregunta1/paso2_preprocesamiento/estudios_coronal.csv \
    --cache pregunta1/paso2_preprocesamiento/cache_coronal
```

Resultado: **574 estudios, 2.76 GB, 0 fallos, 1.1 min**. Verificado que las 574
series elegidas son todas coronales y que **difieren de la sagital en los 550
estudios compartidos** — es decir, el modelo coronal ve imágenes genuinamente
distintas, no un reordenamiento de las mismas.

### 24 estudios quedaron sin serie sagital fluid-sensitive

De los 600 muestreados, 24 (4%) no tienen ninguna serie que cumpla
`Anatomical_Plane == "Sagittal"` y `Fluid_Sensitive == 1` y `Fat_Suppression == 1`.
Se descartan, quedando **576**, dentro del rango 300–600 que pide el examen.

Alternativa no implementada: caer a otro plano para esos 24. Se prefirió mantener
el protocolo homogéneo — mezclar planos sin declararlo contaminaría la ablación
de "elección de secuencia/plano" del entregable #4.

### El `SeriesDescription` no se usa para filtrar

Tiene 21 valores distintos, con `"DummySeriesDesc!"` en el 10% y el resto en
varios idiomas y convenciones (`SAG T2 FS`, `SAG DP SPIR`, `pd_tse_fs_sag_320`).
El filtro usa `Anatomical_Plane`, `Fluid_Sensitive` y `Fat_Suppression` de
`train_series.csv`, que los organizadores ya normalizaron.

### No se usó 3D

El espaciado entre cortes (2.875 mm) es de 6 a 20 veces mayor que el espaciado
dentro del corte (0.146–0.562 mm): el volumen es **muy anisotrópico**, un kernel
3×3×3 mezclaría 1.5 mm en el plano con 8.6 mm entre cortes. Además no hay pesos
preentrenados buenos en 3D. El examen pide 2D o 2.5D, y 2.5D es la opción correcta
por geometría, no solo por el enunciado.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | ~~Variante coronal del caché~~ | ✅ **hecho** — 574 estudios, 2.76 GB |
| 2 | ~~Dataset de PyTorch que lea el caché~~ | ✅ **hecho** — `DatasetRodilla` en [`../paso3_modelos/datos.py`](../paso3_modelos/datos.py) |
| 3 | ~~Partición train/val sobre las etiquetas débiles~~ | ✅ **hecho** — `particionar()` en [`../paso3_modelos/particion.py`](../paso3_modelos/particion.py) |
