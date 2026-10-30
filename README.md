# Pregunta 1 — Clasificación Multietiqueta de Anomalías de Rodilla en MRI

RSNA Knee Abnormality Detection · Examen Parcial 2026-II · UNI FIIS

Entrega: **31 de octubre de 2026**. Penalización 10% por día.

---

## El problema en una frase

Predecir **doce hallazgos clínicos simultáneos** en estudios de MRI de rodilla,
cuando solo **58 de los 4,407 estudios** tienen etiqueta verificada por radiólogo
y el resto solo trae el reporte en texto libre, en nueve idiomas.

## Los siete entregables del examen

| # | Entregable | Paso que lo resuelve |
|---|---|---|
| 1 | Pipeline de extracción de etiquetas débiles, validado contra los dorados, con tasa de ruido | [paso1](paso1_etiquetas_debiles/) |
| 2 | Dos arquitecturas: una CNN desde cero + una con backbone ImageNet, 12 salidas sigmoides | [paso3](paso3_modelos/) |
| 3 | AUC-ROC macro + desglose por etiqueta + precisión/recall/F1 a umbral justificado | [paso4](paso4_evaluacion/) |
| 4 | Ablación sobre al menos dos factores | [paso5](paso5_ablacion_gradcam/) |
| 5 | Grad-CAM para al menos tres hallazgos, con discusión de plausibilidad anatómica | [paso5](paso5_ablacion_gradcam/) |
| 6 | Análisis de desbalance + al menos una mitigación evaluada | [paso5](paso5_ablacion_gradcam/) |
| 7 | Media página sobre sesgo geográfico/lingüístico | [paso1](paso1_etiquetas_debiles/) (los datos) + [paso6](paso6_informe/) (la redacción) |

Más los requisitos transversales: informe de 6–10 páginas en formato NeurIPS con
mínimo 5 referencias, repositorio reproducible, y ejecución de extremo a extremo
con **un único comando**.

## Los seis pasos

```
paso1_etiquetas_debiles/    4,349 reportes en 9 idiomas  →  12 etiquetas binarias
         ↓ (bloquea a todo lo demás: sin etiquetas no hay con qué entrenar ni estratificar)
paso2_preprocesamiento/     ZIP de 570 GB  →  600 estudios  →  tensores 2.5D cacheados
         ↓
paso3_modelos/              Inception propia  +  ResNet-50 ImageNet, condiciones idénticas
         ↓
paso4_evaluacion/           macro-AUC sobre los 56 dorados evaluables, con bootstrap
         ↓
paso5_ablacion_gradcam/     ablación · Grad-CAM · desbalance
         ↓
paso6_informe/              informe NeurIPS + repo reproducible
```

| Paso | Qué produce | Estado |
|---|---|---|
| [1 — Etiquetas débiles](paso1_etiquetas_debiles/) | `etiquetas_debiles.csv` + tabla de acuerdo por etiqueta y por idioma | 🟡 ~90% |
| [2 — Preprocesamiento](paso2_preprocesamiento/) | Caché de tensores 2.5D de 573 estudios | 🟡 ~85% |
| [3 — Modelos](paso3_modelos/) | Las dos arquitecturas entrenadas | 🟡 ~85% |
| [4 — Evaluación](paso4_evaluacion/) | AUC macro + desglose + intervalos bootstrap | 🟡 ~85% |
| [5 — Ablación / Grad-CAM](paso5_ablacion_gradcam/) | 4 factores de ablación, 3 mapas, análisis de desbalance | 🟡 ~80% |
| [6 — Informe](paso6_informe/) | Comando único + material del informe | 🟡 ~40% |

### El comando único que pide el examen

De extremo a extremo, desde un entorno limpio, son tres pasos (crear el entorno,
instalar dependencias, ejecutar el pipeline):

```bash
# 1. Crear y activar un entorno virtual
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate

# 2. Instalar las dependencias de la Pregunta 1
pip install -r pregunta1/requirements.txt
# GPU (CUDA 12.8): reinstalar torch/torchvision desde el indice de PyTorch
# pip install "torch>=2.11" "torchvision>=0.26" --index-url https://download.pytorch.org/whl/cu128

# 3. El comando unico que corre todo el pipeline
python pregunta1/paso6_informe/ejecutar_todo.py
```

`ejecutar_todo.py` es **idempotente**: cada etapa se salta si su artefacto ya
existe, así que una re-ejecución solo rehace lo que falta (`--rehacer` fuerza
todo, `--rapido` usa 3 épocas por modelo para validar el circuito). La ejecución
COMPLETA requiere el dataset RSNA (ver [Dónde están los datos](#dónde-están-los-datos));
sin él, los pasos que leen imágenes no corren, pero el informe se recompila
desde los JSON ya versionados. Ver [paso6](paso6_informe/).

### Resultados medidos

**Entregable #1 — etiquetas débiles**

```
acuerdo macro 74.3%  ·  ruido 25.7%  ·  F1 macro 0.589
mejora por traducir: +0.076  IC95% [+0.039, +0.127]   (bootstrap pareado, n=58)
estudios sin etiqueta: 18.3% (807 de 4 407)
```

Sobre la mitad de los dorados **reservada** —la que no se usó para ajustar el
extractor— el F1 macro es **0.56**, no 0.589; ver
[paso1](paso1_etiquetas_debiles/#8-se-eligió-la-estrategia-midiendo-las-cuatro-opciones).

**Entregables #2 y #3 — las dos arquitecturas sobre 56 dorados**

| | Parámetros | AUC dorados | IC95% |
|---|---|---|---|
| A — Inception desde cero | 0.7M | 0.628 | [0.569, 0.687] |
| B — ResNet-50 ImageNet | 24.06M | 0.646 | [0.579, 0.703] |
| Mejor configuración (B + pooling por media) | 23.53M | **0.708** | [0.662, 0.755] |

Los intervalos de A y B **se solapan**: con n=56 no se puede afirmar que una sea
mejor que la otra.

**El AUC más alto que se midió en toda la pregunta no es ninguno de esos tres:
es 0.7269, del ensamble de las tres corridas** (`paso4_evaluacion/ensamble.json`,
IC95% [0.677, 0.772], promediando probabilidades). Mejora al mejor miembro
individual (0.7217) en **+0.005**, que con este tamaño de muestra no es
distinguible de cero — y por eso el ensamble **no** se reporta como el resultado
de los entregables #2 y #3, que piden las dos arquitecturas. Se deja anotado acá
porque antes no aparecía en ningún README: un número que existe en el repositorio
y no se menciona en ninguna parte es indistinguible de uno que se escondió.

La razón de que gane tan poco está en el mismo JSON: la correlación entre las dos
corridas de ResNet-50 es **0.758**, así que aportan casi la misma información.
Con Inception baja a 0.45-0.50, que es de donde sale la poca ganancia que hay.

**Entregable #4 — ablación: 2 de 7 diferencias son reales**

| Factor | Efecto en AUC dorados |
|---|---|
| **ImageNet → desde cero** | **+0.084**, IC [+0.002, +0.165] — significativo |
| **Plano sagital → coronal** | **+0.056**, IC [+0.005, +0.108] — significativo |
| Pooling: atención → media | +0.062, IC [-0.001, +0.128] — no significativo |
| BCE → focal | inconsistente según el pooling |
| Inception → ResNet-50 | -0.018, no significativo |

**Entregable #5 — Grad-CAM:** Effusion → receso suprapatelar ✅ · ACL → región
intercondílea ✅ · Medial Meniscus → esquina de la imagen ❌

**Entregable #6 — desbalance:** prevalencias de 11.3% a 57.8%; ninguna mitigación
mejora el macro-AUC de forma significativa.

**Estado del código:** la suite corre con `pytest tests/`.

## La regla metodológica que sostiene todo el trabajo

> **Las decisiones se toman con validación cruzada sobre las etiquetas débiles
> (n = 4,349). Los 58 dorados se tocan UNA SOLA VEZ, al final, para el número
> que se reporta.**

Motivo: con 58 estudios, el intervalo de confianza del 95% sobre un AUC ronda
**±0.15**. Para MCL, que tiene 9 positivos, es peor. Si se eligen arquitectura e
hiperparámetros mirando esos 58, el número final no mide generalización, mide
sobreajuste a la validación.

Consecuencia práctica: toda tabla de resultados sobre los dorados lleva
**intervalos por bootstrap**. Una diferencia de 3 puntos de AUC entre dos
modelos, con n=58, no es una diferencia.

---

## Lo que ya sabemos del dataset (verificado, no supuesto)

| | |
|---|---|
| Estudios de entrenamiento | 4,407 |
| Con las 12 etiquetas doradas | **58** |
| Solo con reporte en texto libre | 4,349 |
| Series | 24,371 (mediana 5 por estudio, rango 3–14) |
| Cortes por serie | 19–46, mediana 28 |
| Test real | ~1,300 estudios (los 3 de `test.csv` son solo un ejemplo) |
| Dataset completo | 819,640 archivos · 569.76 GB |

**Idiomas de los reportes** (detectados sobre los 4,407):

```
inglés 39.4%  ·  español 15.5%  ·  turco 12.4%  ·  croata 9.2%  ·  griego 7.3%
alemán 5.9%   ·  búlgaro 5.0%   ·  neerlandés 3.5%  ·  francés 1.8%
```

Dos de ellos sin alfabeto latino (griego, búlgaro). Los 58 dorados están
repartidos en 8 idiomas: 28 en inglés, 10 español, 6 turco, 4 croata, 3 griego,
3 búlgaro, 2 neerlandés, 2 alemán.

**Prevalencia en los 58 dorados** — el desbalance que motiva el entregable #6:

| Hallazgo | Positivos | | Hallazgo | Positivos |
|---|---|---|---|---|
| Effusion | 35 (60%) | | Lateral Meniscus | 23 (40%) |
| Synovitis | 27 (47%) | | PF OA | 21 (36%) |
| Medial Meniscus | 26 (45%) | | Contusion | 19 (33%) |
| ACL | 24 (41%) | | Fracture | 18 (31%) |
| Medial OA | 15 (26%) | | Baker's | 12 (21%) |
| Lateral OA | 11 (19%) | | **MCL** | **9 (16%)** |

**Heterogeneidad técnica de las imágenes** (medida sobre 764 cortes reales):

- 10 resoluciones distintas: de 320×320 a 1024×1024
- `PixelSpacing` de 0.146 a 0.562 mm/px — **factor 4**
- Campo de visión físico de 130 a 240 mm (mediana 170)
- 8 fabricantes, 1.5T y 3T mezclados
- 12 y 16 bits mezclados
- Intensidades sin escala común (la MRI no tiene unidades absolutas)
- `Laterality` ausente en el **54%** de los cortes
- `SeriesDescription` inservible para filtrar: 21 valores distintos, `"DummySeriesDesc!"` en el 10%

Por eso el preprocesamiento recorta en **milímetros** usando `PixelSpacing`, no
en píxeles: sin eso, la misma rodilla llega al modelo a escalas físicas
distintas según el escáner.

## Dónde están los datos

| Qué | Dónde |
|---|---|
| ZIP completo de la competencia | `D:\compartido\rsna-knee-abnormality-detection.zip` (265 GB comprimidos, 569.76 GB dentro) |
| Los 5 CSV de metadatos | `data_kaggle/raw/` |
| DICOM sueltos ya extraídos | `data_kaggle/dicom/train_series/` (764 cortes, 27 series) |

El ZIP **no se descomprime**: `zipfile` lee cualquier entrada por nombre a
**348 MB/s**, así que extraer 600 estudios toma menos de un minuto. No hay que
gastar 570 GB de disco ni volver a bajar nada de Kaggle.

Artefactos ya generados por los pasos:

| Qué | Dónde | Tamaño |
|---|---|---|
| Etiquetas débiles de los 4,407 | `pregunta1/paso1_etiquetas_debiles/etiquetas_debiles.csv` | 1 MB |
| Reportes traducidos al inglés | `pregunta1/paso1_etiquetas_debiles/reportes_traducidos.csv` | 8 MB |
| Muestra estratificada | `pregunta1/paso2_preprocesamiento/estudios_seleccionados.csv` | 130 KB |
| Caché de tensores 2.5D | `pregunta1/paso2_preprocesamiento/cache/*.npy` | 4.3 GB (949 archivos: hay sobrantes de corridas viejas) |

Todos están en `.gitignore`: se regeneran con los comandos documentados en el
README de cada paso.

## Advertencia sobre los notebooks públicos

En `data_kaggle/` hay cuadernos públicos de Kaggle descargados como referencia.
El examen es explícito:

> El plagio de cualquier tipo — incluyendo el de cuadernos públicos —
> resultará en la reprobación automática del curso.

Leerlos para entender el problema y citarlos en "trabajos relacionados" es
correcto. Copiar arquitectura, pipeline o código, no. Además, el notebook de
0.946 declara en su propia primera celda que su autor no escribió nada de ese
código.

## Pesos preentrenados

Los pesos de los modelos entrenados estan publicados como *release* (no caben en el repo por el limite de 100 MB/archivo de GitHub):

**https://github.com/sergiomc12/rnap-2026ii-p1-rodilla-mri/releases/tag/v1.0-pesos**

Cada asset `<corrida>_pesos.pt` va en `pregunta1/paso3_modelos/corridas/<corrida>/pesos.pt`. El modelo de referencia del analisis detallado es `resnet50_promedio`. No es necesario descargarlos para leer el informe ni correr el notebook (que lee los JSON de resultados); solo para reusar un modelo ya entrenado sin reentrenar.

> Nota: `paso6_informe/ejecutar_todo.py` regenera las tablas desde los JSON; una corrida COMPLETA del pipeline requiere el dataset RSNA (~570 GB, se descarga con los scripts) y sobrescribe las tablas refinadas. El PDF incluido ya es el final.
