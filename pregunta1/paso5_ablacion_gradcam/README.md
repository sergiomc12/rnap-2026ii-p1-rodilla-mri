# Paso 5 — Ablación, Grad-CAM y desbalance

Resuelve tres entregables a la vez: **#4 (ablación)**, **#5 (Grad-CAM)** y
**#6 (desbalance de clases)**.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`gradcam.py`](gradcam.py) | `GradCAM`<br>`MapaActivacion`<br>`superponer()` | **Escrito desde cero**: mapas de activación por etiqueta |
| [`generar_figuras.py`](generar_figuras.py) | `main()` | Script ejecutable: produce las figuras de los tres hallazgos |
| [`patron_gradcam.py`](patron_gradcam.py) | `PatronHallazgo`<br>`CasoMapa`<br>`centro_de_masa()` | Corre Grad-CAM sobre varios casos por hallazgo y mide si la activación es consistente |
| [`ablacion_cortes.py`](ablacion_cortes.py) | `Config`<br>`correr(...)`<br>`construir_cache(...)`<br>`entrenar(...)`<br>`validar(...)`<br>`main()` | Script ejecutable: ablación de **cuántos cortes** ve el modelo por estudio (8 a 64 grupos). Hoy `ablacion_cortes.json` solo tiene los cachés construidos: **ninguna de esas configuraciones se entrenó** |
| [`resumir_ablacion_cortes.py`](resumir_ablacion_cortes.py) | `comparar(...)`<br>`main()` | Script ejecutable: lee esa ablación y dice si la diferencia supera el ruido entre pliegues. Sale con error mientras no haya corridas entrenadas, en vez de inventar una tabla vacía |
| [`correr_completo.py`](correr_completo.py) | `nombre_cache(...)`<br>`nombre_corrida(...)`<br>`correr(...)`<br>`entrenar(...)`<br>`main()` | Script ejecutable: entrena las dos arquitecturas sobre los **3 415** estudios en vez de los 573. Escribe `completo.json` |

La ablación no necesita código propio: se corre
[`../paso3_modelos/entrenar.py`](../paso3_modelos/entrenar.py) con distintos flags
y se mide todo con [`../paso4_evaluacion/evaluar.py`](../paso4_evaluacion/evaluar.py).
Esa es la ventaja de que ambas arquitecturas compartan el mismo bucle.

## Comandos

### Correr la ablación completa

```bash
E=".venv/Scripts/python.exe pregunta1/paso3_modelos/entrenar.py --epocas 25"

$E --modelo inception                                      # A, referencia
$E --modelo resnet50                                       # B, referencia
$E --modelo resnet50 --perdida focal --etiqueta resnet50_focal
$E --modelo resnet50 --pooling promedio --etiqueta resnet50_promedio
$E --modelo resnet50 --sin-preentrenar --etiqueta resnet50_sin_imagenet
$E --modelo inception --perdida focal --etiqueta inception_focal

.venv/Scripts/python.exe pregunta1/paso4_evaluacion/evaluar.py
```

### Generar las figuras de Grad-CAM

```bash
.venv/Scripts/python.exe pregunta1/paso5_ablacion_gradcam/generar_figuras.py
.venv/Scripts/python.exe pregunta1/paso5_ablacion_gradcam/generar_figuras.py --corrida resnet50_promedio --hallazgos Effusion ACL Fracture
```

Escribe en `figuras/` un PNG por hallazgo con tres paneles —corte original, mapa
de calor, superposición— más `resumen.json`.

### Usar Grad-CAM desde Python

```python
from pregunta1.paso5_ablacion_gradcam.gradcam import GradCAM, superponer

with GradCAM(modelo) as camara:
    mapa, atencion, probabilidad = camara.generar(tensor_estudio, indice_etiqueta)
    # mapa: (224, 224) en [0, 1]   atencion: (8,)   probabilidad: float
```

El contexto cierra los hooks. Dejarlos registrados haría que cada pasada hacia
adelante del modelo siguiera guardando activaciones para siempre.

---

## Entregable #4 — Ablación sobre cinco factores

El examen pide **al menos dos**. Se hicieron cinco, todos con el mismo bucle,
semilla y datos; solo cambia el factor en estudio.

### Factor 1 — Preentrenamiento en ImageNet

| Configuración | AUC dorados | IC95% |
|---|---|---|
| **ResNet-50 con ImageNet** | **0.6461** | [0.5794, 0.7027] |
| ResNet-50 desde cero | 0.5623 | [0.5055, 0.6179] |

**Bootstrap pareado: +0.084 [+0.002, +0.165], P=98% — SIGNIFICATIVA.**

Es uno de los dos únicos efectos que el test sostiene. La brecha también está en
validación (0.7240 contra 0.6406) y en la velocidad de convergencia: la corrida
sin ImageNet alcanzó su mejor época en la **2** y después solo empeoró, mientras
que con ImageNet la mejor llegó en la 8.

Con 416 estudios de entrenamiento no hay datos para aprender los filtros de bajo
nivel desde cero, y los de ImageNet —aunque vengan de fotografías naturales y no
de resonancias— sirven igual.

### Factor 2 — Pooling: atención vs. media

| Configuración | AUC dorados | IC95% |
|---|---|---|
| **ResNet-50 + media** | **0.7085** | [0.6618, 0.7546] |
| ResNet-50 + atención | 0.6461 | [0.5794, 0.7027] |

**Bootstrap pareado: +0.062 [−0.001, +0.128], P=97% — NO significativa.**

La media queda por delante y gana en el 97% de los remuestreos, pero el intervalo
roza el cero: la diferencia **no se puede afirmar**. La dirección es la esperable
si el módulo de atención sobreajusta —agrega parámetros entrenables sobre 416
estudios cuyas etiquetas tienen 26% de ruido, mientras que una media no tiene
parámetros y no puede sobreajustar— y es consistente también en validación
(0.7616 contra 0.7240). Pero con 56 estudios de prueba, afirmarla sería ir más
lejos de lo que el dato permite.

> **Corrección (5-oct-2026).** Este README decía antes:
> *"Bootstrap pareado: +0.103 [+0.048, +0.159], P=100% — SIGNIFICATIVA. El
> resultado más importante del paso"*. Era incorrecto, y no por un error de
> cálculo: provenía de corridas **no reproducibles**. El entrenamiento en GPU no
> fijaba los kernels de cuDNN, de modo que dos corridas de la misma
> configuración con la misma semilla divergían hasta 0.042 de AUC —más que la
> mayoría de los efectos que esta ablación mide. Ver el
> [README del paso 3](../paso3_modelos/README.md) y
> `verificar_determinismo.py`. Todos los números de esta página provienen de
> corridas verificadas idénticas bit a bit.

No invalida implementar la atención —el examen lo exige y queda implementada,
probada y comparada—, pero la conclusión honesta es que **con este tamaño de
muestra no se puede demostrar que aporte ni que perjudique**.

### Factor 3 — Función de pérdida

| Configuración | AUC dorados | IC95% |
|---|---|---|
| ResNet-50 + atención + BCE | 0.6461 | [0.5794, 0.7027] |
| ResNet-50 + atención + focal | 0.6216 | [0.5711, 0.6728] |
| ResNet-50 + media + BCE | **0.7085** | [0.6618, 0.7546] |
| ResNet-50 + media + focal | 0.6922 | [0.6312, 0.7516] |

**Bootstrap pareado: −0.025 [−0.081, +0.034] con atención, −0.016 [−0.051,
+0.016] con media. Ninguna significativa.**

La pérdida focal empeora levemente en las dos configuraciones en que se probó, lo
que al menos es consistente, pero los dos intervalos cruzan cero. La hipótesis de
partida —que al bajar el peso de los negativos fáciles ayudaría con las etiquetas
raras— no se confirma ni se descarta con 56 estudios de prueba.

### Factor 4 — Arquitectura

| Configuración | AUC dorados | IC95% |
|---|---|---|
| ResNet-50 + atención (ImageNet) | 0.6461 | [0.5794, 0.7027] |
| Inception desde cero + atención | 0.6281 | [0.5693, 0.6866] |

**Bootstrap pareado: −0.018 [−0.098, +0.058], P=34% — NO significativa.**

Las dos arquitecturas del entregable #2 son indistinguibles sobre este conjunto
de prueba. Vale la pena decirlo así y no "ResNet-50 es mejor": el intervalo
abarca 0.156 puntos de AUC, casi un tercio del rango útil de la métrica.

### Factor 5 — Elección de secuencia/plano de MRI

| Configuración | AUC sobre los 54 con ambos planos |
|---|---|
| **ResNet-50 + media, sagital** | **0.715** |
| ResNet-50 + media, coronal | 0.659 |

**Bootstrap pareado: +0.056 [+0.005, +0.108], P=99% — SIGNIFICATIVA.**

El segundo de los dos efectos reales. La comparación se restringe a los **54 de
56** estudios dorados que tienen series en ambos planos: comparar sobre
subconjuntos distintos convertiría una diferencia de plano en una diferencia de
muestra.

Dos arreglos que esta comparación necesitó, y que conviene no perder de vista:

1. **No la calculaba ningún script.** El número aparecía en el informe sin
   pipeline detrás. Ahora la produce `comparar_planos()` en `comparar.py`.
2. **`evaluar.py` evaluaba el modelo coronal con tensores sagitales.** Un modelo
   entrenado en un plano mirando imágenes de otro: el AUC que salía de ahí no
   significaba nada. Cada corrida registra ahora en su `historial.json` con qué
   caché se entrenó, y la evaluación lo respeta.

### Resumen: 2 de 7 diferencias son reales, y las dos son sobre los datos

```bash
.venv/Scripts/python.exe pregunta1/paso4_evaluacion/comparar.py
```

| Comparación | delta | IC95% | P(A mejor) | ¿Real? |
|---|---|---|---|---|
| **ImageNet vs desde cero** | **+0.084** | [+0.002, +0.165] | 98% | ✅ |
| **sagital vs coronal** | **+0.056** | [+0.005, +0.108] | 99% | ✅ |
| media vs atención | +0.062 | [−0.001, +0.128] | 97% | ❌ |
| focal vs BCE (con atención) | −0.025 | [−0.081, +0.034] | 20% | ❌ |
| Inception vs ResNet-50 | −0.018 | [−0.098, +0.058] | 34% | ❌ |
| focal vs BCE (con media) | −0.016 | [−0.051, +0.016] | 16% | ❌ |
| BCE ponderada vs BCE | −0.009 | [−0.047, +0.027] | 31% | ❌ |

**Las dos diferencias que sobreviven al test cambian qué ve el modelo, no cómo
está construido**: de dónde vienen sus pesos iniciales y qué plano de la
resonancia se le muestra. Las cinco que no sobreviven son todas decisiones de
arquitectura o de función de pérdida.

Es coherente con lo que pasó en el paso 1, donde mejorar la extracción de
etiquetas produjo los efectos de mayor magnitud de todo el proyecto. Con 416
estudios de entrenamiento y 26% de ruido en las etiquetas, **lo que se le da al
modelo importa más que cómo está hecho**.

> **Corrección (5-oct-2026).** La tabla anterior daba como significativas
> `media vs atención` y `focal vs BCE (con media)`, y como no significativas
> `ImageNet` y `plano`. Es decir: las cuatro filas que cambiaron de veredicto
> son exactamente las que importan. El motivo está arriba —entrenamiento no
> reproducible— y la lección metodológica es que un bootstrap impecable sobre
> corridas irreproducibles no mide nada: el intervalo describe la variabilidad
> del muestreo de estudios, no la del entorno de ejecución, que acá era mayor.

La fila `sagital vs coronal` además **no la calculaba ningún script** hasta esta
corrección: el número figuraba en el informe sin pipeline detrás. Ahora la
produce `comparar_planos()` en `comparar.py`, restringida a los **54 de 56**
estudios dorados que tienen ambos planos —comparar sobre subconjuntos distintos
convertiría una diferencia de plano en una diferencia de muestra.

---

## Entregable #5 — Grad-CAM

### Cómo se implementó

Ninguno de los cuatro notebooks del curso tiene Grad-CAM — verificado: cero
`register_forward_hook` en todos. Se escribió desde cero siguiendo Selvaraju et al.

El método: gradiente del logit de una etiqueta respecto del último mapa de
características convolucional, promediado por canal para obtener la importancia de
cada canal, combinación de canales con esos pesos, y parte positiva. Lo que
sobrevive es dónde estaba la evidencia de esa etiqueta.

Dos particularidades que agrega esta arquitectura:

- La predicción es **por estudio**, no por grupo de cortes, así que el mapa se
  calcula sobre un grupo elegido. Los pesos de atención dicen en cuál se apoyó el
  modelo. **Los dos análisis se componen**: la atención dice *qué corte*, Grad-CAM
  dice *dónde dentro de ese corte*.
- Hay **doce salidas sigmoides independientes**, así que hay un mapa por etiqueta,
  no uno por estudio.

### Resultados sobre tres hallazgos

Se generaron mapas para **tres** hallazgos —Effusion, Medial Meniscus y ACL—,
que es el mínimo que pide el examen. `generar_figuras.py` los fija en
`HALLAZGOS` y hay tres PNG en `figuras/`.

| hallazgo | probabilidad | figura |
|---|---:|---|
| ACL | 0.8149 | [`gradcam_ACL.png`](figuras/gradcam_ACL.png) |
| Effusion | 0.5764 | [`gradcam_Effusion.png`](figuras/gradcam_Effusion.png) |
| Medial Meniscus | 0.4064 | [`gradcam_Medial_Meniscus.png`](figuras/gradcam_Medial_Meniscus.png) |

Figuras en [`figuras/`](figuras/): tres paneles por hallazgo (corte, mapa,
superposición).

### El caso que falla es el más informativo

El mapa de **Medial Meniscus** se enciende en la esquina de la imagen, **fuera de
la rodilla**, con probabilidad **0.4064**: falso negativo a cualquier umbral por
encima de eso. Es exactamente el fallo que el entregable #5 quiere que se
discuta: el modelo no está mirando el menisco.

**Pero no concuerda con la métrica, y eso es lo interesante.** Una versión
anterior de este README decía que Medial Meniscus tiene *"AUC 0.531, apenas sobre
el azar"*, y que Grad-CAM explicaba ese número malo.
`paso4_evaluacion/detalle_por_etiqueta.json` dice otra cosa: Medial Meniscus
tiene **AUC 0.734 [0.595, 0.861]**, el **quinto mejor de los doce**, y **no cruza
el azar**. El 0.531 es de **Lateral Meniscus**, otra etiqueta.

Así que el fallo es más incómodo de lo que decía: el modelo **acierta** en Medial
Meniscus con un AUC respetable **mientras mira fuera de la rodilla**. Grad-CAM no
está explicando un número malo, está mostrando que un número bueno puede
sostenerse sobre evidencia equivocada. Para una herramienta clínica eso es peor
que un AUC bajo, porque el AUC no lo delata.

### La confianza NO correlaciona con la plausibilidad

Una versión anterior de este README afirmaba que *"el modelo mira bien justo donde
está seguro"*, ordenando **seis** casos por probabilidad (0.90, 0.84, 0.72 arriba;
0.49, 0.33, 0.19 abajo). **Esos seis casos no existen**: hay tres figuras, con
probabilidades 0.8149, 0.5764 y 0.4064. Ninguno de los seis números aparece en
`figuras/resumen.json`.

Medido sobre los 96 mapas de `patron_gradcam.json`, que es donde la pregunta se
puede responder, la relación **no está**:

| relación medida | n | ρ de Spearman |
|---|---:|---:|
| probabilidad del caso vs. ¿el mapa cae en el borde? | 96 mapas | **−0.026** |
| probabilidad media vs. dispersión del hallazgo | 12 hallazgos | **+0.070** |
| AUC vs. dispersión del hallazgo | 12 hallazgos | **+0.545** |

Las dos primeras son cero. La tercera tiene el **signo contrario** al de la
afirmación: los hallazgos con **mejor** AUC son los que miran en lugares **más
dispersos**.

Lo único que sobrevive es una diferencia de medias, débil y sin orden detrás: los
12 mapas que caen en el borde tienen probabilidad media **0.322**, contra
**0.436** de los 84 que caen dentro del cuerpo. Con ρ ≈ 0 eso no alcanza para
decir que la confianza anticipe la plausibilidad, y **acá no se dice**.

La lectura clínica que la versión anterior sacaba de esto —"las predicciones de
baja confianza están además basadas en evidencia equivocada", y por eso un umbral
bajo las convierte en positivos peligrosos— **no se sostiene sobre estos
números**. Lo que sí se sostiene es peor: la evidencia equivocada aparece **a
cualquier nivel de confianza**, así que la probabilidad no sirve para filtrarla.

### El patrón, sobre varios casos por hallazgo

Un solo ejemplo no permite distinguir un patrón de una casualidad. Se corrió
Grad-CAM sobre **hasta 8 estudios dorados positivos de cada hallazgo** (96 mapas
en total) y se midió el **centro de masa** de la región caliente en coordenadas
normalizadas, donde (0,0) es la esquina superior izquierda.

Lo que importa no es la posición absoluta ---la anatomía se desplaza entre
estudios--- sino la **dispersión entre casos**: una desviación chica indica que el
modelo mira siempre el mismo lugar.

```bash
.venv/Scripts/python.exe pregunta1/paso5_ablacion_gradcam/patron_gradcam.py
```

| hallazgo | n | p media | centro medio | dispersión | en borde | ¿consistente? |
|---|---:|---:|---|---:|---:|---|
| **Lateral Meniscus** | 8 | 0.259 | (0.64, 0.53) | 0.152 | 0 | **sí** |
| **ACL** | 8 | 0.503 | (0.53, 0.49) | 0.163 | 0 | **sí** |
| **Effusion** | 8 | 0.684 | (0.46, 0.53) | 0.183 | 0 | **sí** |
| Synovitis | 8 | 0.476 | (0.64, 0.52) | 0.184 | 1 | no |
| MCL | 8 | 0.326 | (0.54, 0.63) | 0.195 | 2 | no |
| Lateral OA | 8 | 0.192 | (0.51, 0.49) | 0.219 | 1 | no |
| Medial Meniscus | 8 | 0.537 | (0.58, 0.49) | 0.247 | 1 | no |
| Contusion | 8 | 0.470 | (0.47, 0.47) | 0.249 | 1 | no |
| Fracture | 8 | 0.303 | (0.58, 0.59) | 0.269 | 2 | no |
| Medial OA | 8 | 0.193 | (0.55, 0.50) | 0.270 | 1 | no |
| PF OA | 8 | 0.601 | (0.44, 0.51) | 0.302 | 2 | no |
| Baker's | 8 | 0.515 | (0.47, 0.46) | 0.316 | 1 | no |

**Solo 3 de 12 hallazgos tienen activación consistente** (dispersión < 0.20 y
ningún caso en el borde): **ACL, Lateral Meniscus y Effusion**, según
`patron_gradcam.json`.

**Corrección.** Una versión anterior de este README concluía que *"el fallo de
menisco medial no era una casualidad"*, con 3 de 8 casos en el borde y la peor
dispersión de las doce (0.399). El JSON no lo respalda: Medial Meniscus tiene
dispersión **0.247** y **1** caso en el borde, y la peor dispersión es la de
**Baker's (0.316)**. Esa conclusión era la más citada de esta sección y **no se
sostiene**.

Lo que sí se sostiene: de los dos hallazgos que la figura mostraba como
plausibles, **ACL (0.163) es el segundo más consistente de los doce**. Pero el
primero es **Lateral Meniscus (0.152)**, que la figura nunca cubrió — o sea que
mirar tres casos no alcanzaba para ordenarlos.

---

## Entregable #6 — Desbalance de clases

### Prevalencia en la muestra usada

| Etiqueta | Positivos de 576 | Prevalencia |
|---|---|---|
| Lateral OA | 73 | 12.2% |
| Medial OA | 84 | 14.0% |
| Fracture | 88 | 14.7% |
| MCL | 102 | 17.0% |
| Synovitis | 102 | 17.0% |
| Contusion | 119 | 19.8% |
| Baker's | 132 | 22.0% |
| ACL | 134 | 22.3% |
| PF OA | 177 | 29.5% |
| Lateral Meniscus | 188 | 31.3% |
| Effusion | 262 | 43.7% |
| Medial Meniscus | 304 | 50.7% |

Factor 4 entre la más rara y la más común.

### Dos mitigaciones implementadas y evaluadas

**1. BCE ponderada por prevalencia** (`pesos_por_prevalencia()`): cada etiqueta
recibe un peso positivo inversamente proporcional a su prevalencia, con tope en
10× para que una etiqueta casi ausente no desestabilice el entrenamiento.

**2. Pérdida focal** (`PerdidaFocal`): multiplica cada término por `(1 - p_t)^γ`,
reduciendo el peso de los ejemplos que el modelo ya acierta. Verificado: con
γ=2 y α=0.25 reduce **880×** la contribución de un ejemplo fácil y solo **2.2×** la
de uno difícil.

### Resultado sobre las etiquetas raras

| Etiqueta | BCE | BCE ponderada | focal |
|---|---|---|---|
| Lateral OA (12.2%) | 0.693 | 0.630 | **0.711** |
| Medial OA (14.0%) | 0.816 | **0.841** | 0.816 |
| Fracture (14.7%) | **0.813** | 0.676 | 0.736 |
| MCL (17.0%) | 0.638 | 0.598 | **0.641** |
| **MACRO** | **0.708** | 0.699 | 0.692 |

**Ninguna mitigación mejora el macro-AUC**, y el test pareado lo confirma:

| Mitigación | delta vs BCE | IC95% | ¿Significativa? |
|---|---|---|---|
| BCE ponderada | −0.009 | [−0.047, +0.027] | no (P=31%) |
| Pérdida focal | −0.016 | [−0.051, +0.016] | no (P=16%) |

**Ninguna de las dos mitigaciones mueve la aguja de forma medible.** Una versión
anterior de este README afirmaba que la focal empeoraba de forma *significativa*
(−0.035, IC [−0.067, −0.003]); `comparaciones.json` dice otra cosa: −0.016 con
IC [−0.051, +0.016], que **cruza el cero**. La tabla generada
(`informe/tablas/ablacion.tex`) ya imprimía el valor correcto, así que el error
estaba solo en la prosa copiada a mano.

Por etiqueta rara el resultado está
repartido: la ponderada ayuda a MCL, la focal a Medial OA, y BCE simple gana en
Lateral OA y Fracture.

La lectura honesta: **con 56 estudios de prueba, las diferencias por etiqueta están
dentro del ruido de muestreo**. La mitigación del desbalance no es la palanca
dominante en este problema; el 26.0% de ruido en las etiquetas sí lo es.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | Grad-CAM sobre un modelo con atención, comparando los pesos | El mejor modelo usa media, cuyos pesos son uniformes; la composición "qué corte + dónde" solo se puede mostrar con atención |
| 2 | ~~Más de un caso por hallazgo~~ | ✅ **hecho** — 8 casos × 12 hallazgos, con medición de consistencia |
| 3 | Grad-CAM sobre un falso positivo | Mostraría qué está mirando el modelo cuando se equivoca hacia el otro lado |
| 4 | Sobremuestreo de positivos como tercera mitigación | Es ortogonal a cambiar la pérdida |
