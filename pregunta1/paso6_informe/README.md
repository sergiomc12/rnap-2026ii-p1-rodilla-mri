# Paso 6 — Informe y repositorio reproducible

Resuelve el **entregable #7** y los requisitos de entrega.

> Informe (PDF, **formato NeurIPS**) + enlace al repositorio de GitHub. El código
> debe ejecutarse de extremo a extremo con **un único comando** documentado en el
> README, sobre una especificación de entorno (`requirements.txt` o
> `environment.yml`).

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`ejecutar_todo.py`](ejecutar_todo.py) | `main()`<br>`Etapa` | **El comando único.** Corre las 26 etapas en orden, saltando lo que ya existe |
| [`generar_tablas.py`](generar_tablas.py) | `main()` | Genera las tablas del informe desde los JSON. Es la etapa final |
| [`compilar_informe.py`](compilar_informe.py) | `compilar()` | `latexmk`, o la secuencia pdflatex/bibtex/pdflatex×2 |
| | `paginas_del_log()` | Las páginas del PDF, leídas del log de LaTeX |
| | `referencias_del_bbl()` | Cuenta las referencias **citadas**, no las del `.bib` |
| | `paginas_de_cuerpo()` | **Las páginas de cuerpo**, que son las que limita el enunciado. Busca la página donde arranca la bibliografía y mide la altura de su título: contar páginas del PDF obligaría a suponer cuánto ocupan las referencias |
| | `_altura_relativa()` | Qué fracción de esa página ocupa el cuerpo. El corte es el **título** de la bibliografía, no su entrada `[1]`: si la página arranca con *References*, el cuerpo terminó en la anterior |
| | `errores_del_log()` | Las líneas de error, para no abrir el log |
| [`exportar_word.py`](exportar_word.py) | `main()` | Genera el `.docx` con pandoc desde el **mismo** `.tex` que el PDF |
| | `cuerpo_plano()` | Recorta el cuerpo y resuelve los `\input{}`. Pasarle el `.tex` entero a pandoc **se cuelga** con `neurips.sty`, y los `\input` no los sigue: sin esto las tablas faltarían en el Word sin ningún aviso |
| | `metadatos()` | Rescata título y autores del preámbulo descartado; sin esto el Word arrancaba directo en el resumen |

## Comandos

### El comando único que pide el examen

```bash
.venv/Scripts/python.exe pregunta1/paso6_informe/ejecutar_todo.py
```

Es **idempotente**: cada etapa se salta si su artefacto ya existe. Una corrida
limpia desde cero toma ~70 min, dominada por la traducción (~45 min).

```bash
# verificar el circuito completo con 3 epocas por modelo
.venv/Scripts/python.exe pregunta1/paso6_informe/ejecutar_todo.py --rapido

# rehacer todo, ignorando los artefactos existentes
.venv/Scripts/python.exe pregunta1/paso6_informe/ejecutar_todo.py --rehacer

# correr solo un tramo
.venv/Scripts/python.exe pregunta1/paso6_informe/ejecutar_todo.py --desde 3 --hasta 5
```

### Las etapas

| # | Paso | Etapa | Artefacto |
|---|---|---|---|
| 1 | 1 | Traducir reportes no-ingleses | `reportes_traducidos.csv` |
| 2 | 1 | Extraer etiquetas débiles | `etiquetas_debiles.csv` |
| 3 | 1 | Medir el extractor contra los dorados | *(solo imprime)* |
| 4 | 2 | Elegir la muestra estratificada | `estudios_seleccionados.csv` |
| 5 | 2 | Preprocesar y cachear tensores | `cache/*.npy` |
| 6 | 3 | Entrenar A: Inception desde cero | `corridas/inception_bce_atencion/` |
| 7 | 3 | Entrenar B: ResNet-50 ImageNet | `corridas/resnet50_bce_atencion/` |
| 8 | 5 | Ablación: pooling por media | `corridas/resnet50_promedio/` |
| 9 | 5 | Ablación: pérdida focal | `corridas/resnet50_focal/` |
| 10 | 5 | Ablación: sin pesos de ImageNet | `corridas/resnet50_sin_imagenet/` |
| 11 | 6 | Desbalance: BCE ponderada | `corridas/resnet50_promedio_ponderada/` |
| 12 | 4 | Evaluar sobre los dorados | `resultados.json` |
| 13 | 5 | Figuras de Grad-CAM | `figuras/*.png` |

## Requisitos previos

| | |
|---|---|
| ZIP de la competencia | `D:\compartido\rsna-knee-abnormality-detection.zip` |
| Entorno | `pip install -e ".[dev]"` + `torch` con CUDA |
| GPU | Opcional; en CPU el entrenamiento es mucho más lento |

---

## El informe

Vive en [`informe/`](informe/) y **compila**:

```bash
cd pregunta1/paso6_informe/informe
latexmk -pdf informe.tex
```

| Archivo | Qué es |
|---|---|
| [`informe.tex`](informe/informe.tex) | El informe, con **todos los resultados medidos ya cargados** |
| [`referencias.bib`](informe/referencias.bib) | 12 entradas, cada una comentada con **qué archivo del repo la implementa** |
| [`neurips.sty`](informe/neurips.sty) | Formato NeurIPS reimplementado |

**Sobre la plantilla:** la hoja de estilo oficial se distribuye desde el sitio de
la conferencia y no está en CTAN; su descarga devuelve 403. `neurips.sty`
reproduce lo que define la guía: papel carta, caja de 5.5 × 9 pulgadas, cuerpo de
10 pt Times, título de 17 pt, resumen sangrado a ambos lados, encabezados de
sección de 12 pt y referencias en cuerpo chico. Si el grupo consigue la oficial,
basta reemplazar el archivo.

## Estado

| | |
|---|---:|
| cuerpo | **9.77 páginas** (límite 6–10) |
| páginas del PDF | 11, con las referencias aparte |
| referencias citadas | **12** (mínimo 5) |
| errores de LaTeX / citas sin resolver | 0 / 0 |
| tablas / figuras | 5 / 4 |
| Word | `informe.docx`, 6 120 palabras |

Todo esto lo verifica `compilar_informe.py` en cada corrida, contra el log de
LaTeX, el `.bbl` y el PDF —no contra la vista previa—. El límite de páginas es
sobre el **cuerpo**: se mide dónde arranca la bibliografía, no se supone cuánto
ocupa.

Cero `% TODO`: todas las secciones están redactadas, incluidas motivación,
trabajos relacionados, conclusiones y la interpretación en prosa de la ablación y
de las figuras de Grad-CAM.

## Material para el informe

### Estructura que pide el examen

> motivación · trabajos relacionados (**mínimo 5 referencias**) · metodología ·
> experimentos · resultados · **discusión crítica de las limitaciones**

### Referencias disponibles en `doc/`

Las cinco de arquitectura están cubiertas sin buscar nada más:

1. LeCun et al. — *Gradient-Based Learning Applied to Document Recognition* (`lecun-98.pdf`)
2. Krizhevsky et al. — AlexNet (`AlexNetPaper.pdf`)
3. Simonyan & Zisserman — VGG (`1409.1556v6.pdf`)
4. Szegedy et al. — GoogLeNet / Inception (`GooGleNet.pdf`) ← **base de la arquitectura A**
5. He et al. — ResNet (`ResNet.pdf`) ← **base de la arquitectura B**

Faltan citar, y no están en `doc/`:

- Selvaraju et al. — Grad-CAM (base de la implementación del paso 5)
- Ilse et al. — Attention-based Deep Multiple Instance Learning (base del pooling)
- Lin et al. — Focal Loss
- Chapman et al. — NegEx (base del detector de negación del paso 1)
- NLLB Team — No Language Left Behind (el traductor del paso 1)

### Resultados a reportar

**Entregable #1 — etiquetas débiles**

```
acuerdo macro : 74.3%        tasa de ruido : 25.7%
F1 macro      : 0.589        (desde 0.512 sin traducir, +0.076 IC95% [+0.039, +0.127])
F1 macro      : 0.56         sobre la mitad RESERVADA de los dorados
estudios sin ninguna etiqueta : 18.3%  (807 de 4 407)
```

**Entregable #2 y #3 — las dos arquitecturas sobre 56 dorados**

| | AUC dorados | IC95% |
|---|---|---|
| Inception desde cero (0.7M) | 0.628 | [0.569, 0.687] |
| ResNet-50 ImageNet (24.06M) | 0.646 | [0.579, 0.703] |
| **Mejor configuración** (ResNet-50 + media) | **0.708** | [0.662, 0.755] |

**Entregable #4 — ablación sobre cuatro factores**

| Factor | Efecto |
|---|---|
| **ImageNet → desde cero** | **+0.084** ← significativo |
| **Plano sagital → coronal** | **+0.056** ← significativo |
| Pooling atención → media | +0.062, IC roza cero — no significativo |
| ImageNet → desde cero | −0.033 |
| BCE → focal | -0.025 con atención, -0.016 con media — ninguna significativa |
| Inception → ResNet-50 | -0.018, no significativo |

**Entregable #5 — Grad-CAM:** dos mapas anatómicamente plausibles (Effusion →
receso suprapatelar, ACL → región intercondílea) y uno que falla (Medial Meniscus
→ esquina de la imagen, fuera de la articulación).

**Entregable #6 — desbalance:** prevalencias de 12.2% a 50.7%; ninguna mitigación
mejora el macro-AUC de forma significativa.

### Entregable #7 — sesgo geográfico y lingüístico (media página)

Todo el material está medido en el paso 1. Los puntos a desarrollar:

**La calidad de la extracción varía por idioma, y mucho.** Antes de traducir, el
F1 macro iba de 0.66 en inglés a **0.00** en croata, griego y búlgaro. El
extractor funcionaba para el 55% de los estudios y producía ruido puro para el
45%, correlacionado directamente con el país del centro emisor.

**Traducir mitiga pero no iguala.** Tras la traducción híbrida, el griego sigue al
**54.2% de estudios sin ninguna etiqueta** contra 11.5% en inglés. La traducción
médica de NLLB degrada terminología: se observó *"Left **eye** MRG"* por knee,
*"**cross-linked** ligaments"* por cruciate, *"minimal **urinary** growth"* por
effusion.

**Y degrada lo que ya funcionaba.** El español nativo daba F1 0.48; traducido cayó
a 0.29. Por eso la estrategia final no traduce inglés ni español.

**Implicancia clínica.** Un modelo entrenado con estas etiquetas aprende
principalmente de estudios cuyo reporte estaba en inglés o español. Desplegado en
un centro que reporta en griego o búlgaro, su rendimiento real sería peor que el
informado — y nada en la métrica agregada lo advertiría. Un AUC macro de 0.708
oculta que la supervisión de entrenamiento fue sistemáticamente peor para unos
centros que para otros.

**El límite no verificable.** De los 4,349 estudios sin etiqueta dorada, 1,989
están en los siete idiomas problemáticos. Su calidad de etiqueta **no se puede
medir**: no existe verdad para ellos. Lo que se reporta se estima sobre los 30
dorados no-ingleses, y se extrapola.

### Discusión crítica — lo que hay que declarar

1. **n = 56.** El IC95% de un AUC ronda ±0.06 en el macro y es mucho peor por
   etiqueta. MCL se estima sobre **9 positivos**.
2. **Las dos arquitecturas no son distinguibles**: 0.628 vs 0.646 con intervalos
   solapados.
3. **El ranking se invierte** entre validación (etiquetas débiles) y dorados. Si
   se hubiera reportado el número de validación, la conclusión sería la contraria.
4. **26.0% de las etiquetas de entrenamiento están mal.** Es el techo real del
   problema, por encima de cualquier decisión de arquitectura.
5. **Las etiquetas doradas no son transcripción literal del reporte**: hay criterio
   clínico. Un estudio dice *"Moderate joint effusion"* y su etiqueta de Effusion
   es 0. Ningún método basado en texto puede llegar al 100%.
6. **El pooling por atención no se puede afirmar que ayude ni que perjudique**
   (corregido el 5-oct-2026: antes se afirmaba que empeoraba de forma
   significativa, sobre corridas no reproducibles) pese a ser la elección teóricamente
   correcta. Con 416 estudios de entrenamiento, sus parámetros extra sobreajustan.
7. **Un solo ejemplo por hallazgo en Grad-CAM.** No alcanza para distinguir un
   patrón de una casualidad.
8. **ResNet-50 sobreajusta severamente**: train 0.019 contra val 0.567.

---

## Pendiente

| # | Qué falta | Estado |
|---|---|---|
| 1 | ~~Redactar el informe~~ | ✅ **hecho** — 9.77 páginas de cuerpo, cero `% TODO`, 4 figuras, 5 tablas |
| 2 | ~~README raíz con el comando único~~ | ✅ **hecho** |
| 3 | ~~Figuras y tablas~~ | ✅ **hecho** — Grad-CAM, curvas de entrenamiento y las tablas desde los JSON |
| 4 | ~~Exportar el informe a Word~~ | ✅ **hecho** — `exportar_word.py`, etapa del comando único |
| 5 | ~~Citar Grad-CAM, MIL por atención, focal loss, NegEx y NLLB~~ | ✅ **hecho** |
| 6 | Verificar `ejecutar_todo.py --rehacer` desde cero | ⏳ ~70 min desatendidos; validaría que nada dependa de un artefacto viejo |
