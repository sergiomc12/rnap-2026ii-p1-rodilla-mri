# Paso 4 — Evaluación con la métrica oficial

Resuelve el **entregable #3**:

> Evaluación con la métrica oficial de la competencia (**AUC-ROC macro-promediado**
> sobre las doce etiquetas) y **desglose por etiqueta**, usando el conjunto de
> validación de etiqueta dorada. Incluir además **precisión, recall y F1 a un
> umbral elegido y justificado**.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`metricas.py`](metricas.py) | `auc_roc()`<br>`auc_macro()`<br>`ResultadoAUC` | AUC por etiqueta y macro-promedio, calculado por rangos |
| | `bootstrap_auc_macro()`<br>`IntervaloBootstrap` | Intervalo de confianza por remuestreo — **obligatorio con n=56** |
| | `metricas_en_umbral()`<br>`mejor_umbral()`<br>`MetricasUmbral` | Precisión/recall/F1 y elección justificada del umbral |
| [`evaluar.py`](evaluar.py) | `main()`<br>`evaluar_corrida()`<br>`predecir()` | Script ejecutable: evalúa todas las corridas sobre los dorados |
| [`comparar.py`](comparar.py) | `comparar()`<br>`Comparacion` | **Bootstrap pareado**: decide si la diferencia entre dos corridas es real |
| [`detalle_por_etiqueta.py`](detalle_por_etiqueta.py) | `detallar()`<br>`DetalleEtiqueta`<br>`bootstrap_etiqueta()` | IC y matriz de confusión **por etiqueta** |
| [`ensamblar.py`](ensamblar.py) | `a_rangos(...)`<br>`correlacion_media(...)`<br>`main()` | Script ejecutable: ¿promediar varios modelos mejora el AUC? Da **0.7269**, el más alto de la pregunta, con +0.005 sobre el mejor miembro — no distinguible de cero. Escribe `ensamble.json` |

El paso 3 importa `metricas.py` para vigilar el AUC entre épocas.

## Comandos

```bash
.venv/Scripts/python.exe pregunta1/paso4_evaluacion/evaluar.py
.venv/Scripts/python.exe pregunta1/paso4_evaluacion/evaluar.py --corrida resnet50_promedio
```

Escribe `resultados.json` con el AUC macro, el intervalo, el desglose por etiqueta
y las métricas al umbral de cada corrida.

### Usar las clases desde Python

```python
from pregunta1.paso4_evaluacion.metricas import auc_macro, bootstrap_auc_macro, mejor_umbral

resultado = auc_macro(probabilidades, verdad)   # (N, 12) ambos
resultado.macro                                  # promedio sobre las definidas
resultado.por_etiqueta["MCL"]                    # None si la etiqueta no tiene ambas clases

print(bootstrap_auc_macro(probabilidades, verdad, remuestreos=2000))
# 0.708 [0.662, 0.755]

mejor_umbral(prob_validacion, verdad_validacion)  # se elige en validacion, nunca en los dorados
```

---

## Lo que se hizo

### La regla que se respetó

> Las decisiones se toman con las **etiquetas débiles**. Los dorados se tocan
> **una sola vez**, al final.

Arquitectura, pérdida, pooling, épocas y **el umbral** se eligieron sobre los 104
estudios de validación. Lo que este paso reporta es generalización, no un efecto
de selección.

### Resultados sobre los 56 estudios dorados

Todas las corridas son **deterministas y verificadas**: dos ejecuciones de la
misma configuración dan el mismo resultado bit a bit (ver
[paso 3](../paso3_modelos/README.md)).

| Corrida | AUC val | AUC oro | IC95% | F1 | umbral |
|---|---|---|---|---|---|
| **resnet50_promedio** | 0.7616 | **0.7085** | [0.6618, 0.7546] | 0.560 | 0.20 |
| resnet50_promedio_ponderada | 0.7582 | 0.6992 | [0.6480, 0.7490] | 0.517 | 0.25 |
| resnet50_promedio_focal | 0.7449 | 0.6922 | [0.6312, 0.7516] | 0.516 | 0.20 |
| resnet50_promedio_coronal | 0.7490 | 0.6502 | [0.5908, 0.7114] | 0.493 | 0.15 |
| **resnet50_bce_atencion** | 0.7240 | **0.6461** | [0.5794, 0.7027] | 0.490 | 0.25 |
| **inception_bce_atencion** | 0.6978 | **0.6281** | [0.5693, 0.6866] | 0.484 | 0.25 |
| resnet50_focal | 0.7245 | 0.6216 | [0.5711, 0.6728] | 0.477 | 0.25 |
| inception_focal | 0.6513 | 0.5900 | [0.5324, 0.6424] | 0.506 | 0.25 |
| resnet50_sin_imagenet | 0.6406 | 0.5623 | [0.5055, 0.6179] | 0.507 | 0.10 |

### AUC por etiqueta de las dos arquitecturas del entregable #2, más la mejor

| Etiqueta | Inception (desde cero) | ResNet-50 (ImageNet) | ResNet-50 + media |
|---|---|---|---|
| ACL | 0.714 | 0.706 | 0.776 |
| MCL | 0.496 | 0.520 | 0.638 |
| Medial Meniscus | 0.611 | 0.680 | 0.734 |
| Lateral Meniscus | 0.545 | 0.497 | 0.531 |
| Medial OA | 0.680 | 0.720 | 0.816 |
| Lateral OA | 0.610 | 0.655 | 0.693 |
| PF OA | 0.589 | 0.618 | 0.704 |
| Effusion | 0.749 | 0.729 | 0.619 |
| Synovitis | 0.697 | 0.648 | 0.672 |
| Baker's | 0.648 | 0.640 | 0.731 |
| Contusion | 0.594 | 0.652 | 0.773 |
| Fracture | 0.605 | 0.689 | 0.813 |
| **MACRO** | **0.6281** | **0.6461** | **0.7085** |

### El umbral, justificado

Se barren umbrales de 0.05 a 0.95 en pasos de 0.05 **sobre el conjunto de
validación**, y se elige el que maximiza el F1 macro. Luego se aplica tal cual a
los dorados.

Elegirlo sobre los dorados inflaría el F1 que se reporta, ya que se estaría
optimizando contra los mismos datos sobre los que se informa.

Los umbrales elegidos caen entre 0.10 y 0.30, **muy por debajo de 0.5**. Es
consistente con el desbalance: con prevalencias de 12% a 51%, un umbral de 0.5
descarta casi todos los positivos de las etiquetas raras.

### Detalle por etiqueta: el macro esconde dónde falla el modelo

```bash
.venv/Scripts/python.exe pregunta1/paso4_evaluacion/detalle_por_etiqueta.py
```

Sobre la mejor configuración (`resnet50_promedio`), al umbral 0.20
elegido en validación:

| Etiqueta | pos | AUC | IC95% | ancho | TP | FP | FN | TN | F1 |
|---|---|---|---|---|---|---|---|---|---|
| ACL | 23 | 0.776 | [0.639, 0.893] | 0.254 | 20 | 17 | 3 | 16 | 0.667 |
| **MCL** | 9 | 0.638 | **[0.442, 0.828]** | 0.386 | 6 | 20 | 3 | 27 | 0.343 |
| Medial Meniscus | 24 | 0.734 | [0.595, 0.861] | 0.266 | 22 | 18 | 2 | 14 | 0.687 |
| **Lateral Meniscus** | 23 | 0.531 | **[0.378, 0.687]** | 0.310 | 16 | 18 | 7 | 15 | 0.561 |
| Medial OA | 15 | 0.816 | [0.677, 0.928] | 0.252 | 8 | 4 | 7 | 37 | 0.593 |
| Lateral OA | 11 | 0.693 | [0.512, 0.865] | 0.354 | 5 | 9 | 6 | 36 | 0.400 |
| PF OA | 20 | 0.704 | [0.554, 0.840] | 0.285 | 19 | 31 | 1 | 5 | 0.543 |
| **Effusion** | 34 | 0.619 | **[0.461, 0.769]** | 0.308 | 31 | 19 | 3 | 3 | 0.738 |
| Synovitis | 25 | 0.672 | [0.517, 0.808] | 0.291 | 17 | 16 | 8 | 15 | 0.586 |
| Baker's | 11 | 0.731 | [0.511, 0.913] | 0.403 | 6 | 12 | 5 | 33 | 0.414 |
| Contusion | 18 | 0.773 | [0.641, 0.896] | 0.255 | 17 | 25 | 1 | 13 | 0.567 |
| Fracture | 17 | 0.813 | [0.667, 0.927] | 0.260 | 13 | 12 | 4 | 27 | 0.619 |

**3 de las doce etiquetas tienen un intervalo que incluye 0.5**, es
decir, no se distinguen del azar: MCL, Lateral Meniscus, Effusion.

**El macro es 3.3 veces más estrecho que una etiqueta
típica.** Ancho medio por etiqueta: 0.302; ancho del macro:
0.093. Promediar doce estimaciones ruidosas cancela parte del ruido,
de modo que reportar solo el macro sugiere una precisión que las etiquetas
sueltas no tienen.

Los dos casos extremos fallan por razones distintas, y conviene no mezclarlas.
**Lateral Meniscus** tiene 23 positivos —suficientes— y aun
así da 0.531: el modelo no lo detecta.
**Baker's**, con 11 positivos, tiene un
intervalo de 0.403 puntos: ahí el problema
no es que falle, es que no hay datos para saber si falla.

El bootstrap por etiqueta descarta los remuestreos donde la etiqueta queda con una
sola clase: contarlos como 0.5 estrecharía el intervalo artificialmente hacia el
azar.

---

## Hallazgos que condicionan la lectura de todo el proyecto

### El ranking YA NO se invierte entre validación y dorados

```
validación (etiquetas débiles): ResNet-50 0.7240  >  Inception 0.6978
dorados    (radiólogos)       : ResNet-50 0.6461  >  Inception 0.6281
```

> **Corrección (5-oct-2026).** Esta sección afirmaba lo contrario: que el orden
> entre las dos arquitecturas se invertía entre validación y dorados, y que eso
> era *"la justificación empírica más fuerte de la regla metodológica"* de no
> elegir modelo contra los estudios dorados. **Con el entrenamiento
> reproducible, la inversión no ocurre.** Era un artefacto de dos corridas no
> deterministas, del mismo tamaño que la diferencia que pretendía ilustrar.

La regla metodológica sigue en pie —elegir el modelo contra los dorados sería
medir sobre los mismos datos que se reportan—, pero se sostiene por su propio
fundamento, no por esta observación. Lo que sí queda es un hallazgo de segundo
orden más incómodo: cuando el entrenamiento no es reproducible, no solo las
comparaciones numéricas quedan contaminadas, también las observaciones
cualitativas que uno construye sobre ellas.

Lo que sí se mantiene es la **caída de validación a dorados**: toda corrida
pierde entre 0.03 y 0.08 de AUC al pasar de las etiquetas débiles a las
verificadas. Las etiquetas de entrenamiento tienen 26% de ruido, así que una
parte de lo que el modelo aprende es a reproducir los sesgos del extractor.

### La diferencia entre las dos arquitecturas no es significativa

```
Inception  0.6281  IC95% [0.5693, 0.6866]
ResNet-50  0.6461  IC95% [0.5794, 0.7027]
```

Los intervalos **se solapan casi por completo**, y el bootstrap pareado lo
confirma: **delta -0.018, IC95% [-0.098, +0.058], P=34%**.
Con 56 estudios no se puede afirmar que una arquitectura sea mejor que la otra.

```bash
.venv/Scripts/python.exe pregunta1/paso4_evaluacion/comparar.py
```

El pareado es el test correcto acá: ambos modelos se evalúan sobre **los mismos**
56 estudios, así que sus errores están correlacionados y comparar dos intervalos
independientes sería demasiado conservador.

### Todos los modelos caen de validación a dorados

De 0.70–0.76 a 0.58–0.71. Dos causas que se suman:

1. **Las etiquetas de validación son ruidosas** (26.0%), así que ese número está
   inflado respecto de la verdad clínica.
2. **Los dorados son 56 estudios** y no hay ninguna garantía de que su
   distribución coincida con la de la muestra de entrenamiento.

### 3 etiquetas no se distinguen del azar

- **MCL** (9 positivos): AUC 0.638, IC95% [0.442, 0.828]
- **Lateral Meniscus** (23 positivos): AUC 0.531, IC95% [0.378, 0.687]
- **Effusion** (34 positivos): AUC 0.619, IC95% [0.461, 0.769]

Los tres intervalos incluyen 0.5. No es lo mismo en los tres casos: con pocos
positivos el intervalo es ancho y no se puede concluir nada, mientras que con
veintitantos positivos y un AUC cerca de 0.5 lo que se concluye es que el modelo
no aprendió esa etiqueta.

### El AUC macro ignora las etiquetas indefinidas

`auc_roc()` devuelve `None` cuando una etiqueta tiene una sola clase en el
remuestreo, y el macro-promedio solo usa las definidas. Devolver 0.5 arrastraría
la media hacia el azar sin avisar.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | ~~Bootstrap pareado~~ | ✅ **hecho** — ver `comparar.py`; solo 2 de 6 diferencias resultaron reales |
| 2 | ~~Matriz de confusión por etiqueta~~ | ✅ **hecho** — `detalle_por_etiqueta.py` |
| 3 | ~~IC por etiqueta~~ | ✅ **hecho** — 4 de 12 etiquetas no se distinguen del azar |
