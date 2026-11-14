# Guía breve de defensa oral — TP1 Cliente

**Materia:** Inteligencia Artificial — 2C — 2026 · **Profesor:** Parkinson, Christian Anibal
**Equipo 8:** Massi, Franco (L.U. 1156946) · Nari, Agustin Ignacio (L.U. 1165816) · Vanore, Miqueas (L.U. 1179980)

## Mensaje de apertura (30 segundos)

“Clasificamos la baja de 7.043 clientes con tres familias: logística, bosques y redes multicapa. Reservamos una prueba independiente antes de explorar, elegimos por validación cruzada y recién después medimos el resultado final. Recomendamos Logistica, con F1 macro de prueba 0.7376. El archivo no permite verificar anticipación futura.”

## Presentación grupal sugerida (10 minutos)

| Tiempo | Contenido | Apoyo |
|---|---|---|
| 0–2 min | Problema, objetivo, origen y variables | Notebook §§1–2 |
| 2–4 min | Vacíos, EDA y pipeline | EDA y esquema de preprocesamiento |
| 4–6 min | Tres familias, dos MLP y backprop | Notebook §§4 y 6 |
| 6–8 min | CV, baseline y métricas finales | Tablas y matrices |
| 8–10 min | Elección, límites y cierre | Conclusión y casos de error |

Distribuir bloques según el número de integrantes. Todos deben poder explicar el pipeline y las métricas. Abrir la notebook ya ejecutada; no depender de entrenar en vivo ni de Internet. La guía y la notebook son el soporte solicitado, sin necesidad de crear una aplicación.

## Cifras que debemos poder explicar

- 7.043 clientes, 21 columnas originales; 19 predictores, ID excluido y un objetivo.
- 11 TotalCharges vacíos, todos con tenure=0; se convierten a ausencia y se imputan con mediana de entrenamiento.
- 5.634 clientes de desarrollo y 1.409 de prueba; CV estratificada de 3 folds, semilla 42.
- 8 configuraciones × 3 folds = 24 ajustes de CV, más 4 ajustes finales.
- Selección: Logistica, F1 macro CV 0.7325 ± 0.0115.
- Prueba del seleccionado: accuracy 0.8055, precision churn 0.6572, recall 0.5588, F1 churn 0.6040.
- Matriz: TN 926, FP 109, FN 165, TP 209. Detectamos 209 bajas y omitimos 165; 109 alertas fueron falsas.

## Variables del dataset (qué mide cada una)

Un único dataset (una tabla); la unidad es el cliente. Cada fila: 20 atributos + la etiqueta `Churn`.

| Variable | Qué mide / indica en el cliente | Cómo se guarda | Tipo |
|---|---|---|---|
| customerID | Código único del cliente | texto | Nominal (ID) — excluida |
| gender | Género registrado | `Male`/`Female` | Nominal binaria |
| SeniorCitizen | Si es adulto mayor (1) o no (0) | `0`/`1` | Binaria (guardada como número) |
| Partner | Si tiene pareja | `Yes`/`No` | Nominal binaria |
| Dependents | Si tiene personas a cargo | `Yes`/`No` | Nominal binaria |
| tenure | Meses que lleva en la empresa | `0`–`72` | Numérica discreta |
| PhoneService | Si tiene servicio telefónico | `Yes`/`No` | Nominal binaria |
| MultipleLines | Si tiene más de una línea (o sin telefonía) | 3 valores | Nominal (3) |
| InternetService | Tipo de internet contratado | `DSL`/`Fiber optic`/`No` | Nominal (3) |
| OnlineSecurity | Si contrató seguridad online | 3 valores | Nominal (3) |
| OnlineBackup | Si contrató backup online | 3 valores | Nominal (3) |
| DeviceProtection | Si contrató protección de dispositivos | 3 valores | Nominal (3) |
| TechSupport | Si contrató soporte técnico | 3 valores | Nominal (3) |
| StreamingTV | Si usa streaming de TV | 3 valores | Nominal (3) |
| StreamingMovies | Si usa streaming de películas | 3 valores | Nominal (3) |
| Contract | Plazo del contrato | mensual/anual/bianual | Ordinal |
| PaperlessBilling | Si recibe factura electrónica | `Yes`/`No` | Nominal binaria |
| PaymentMethod | Medio de pago | 4 valores | Nominal (4) |
| MonthlyCharges | Importe facturado por mes | `18,25`–`118,75` | Numérica continua |
| TotalCharges | Importe total acumulado | `18,80`–`8.684,80` (11 vacíos) | Numérica continua |
| Churn | Si se dio de baja | `Yes`/`No` | Binaria (objetivo) |

`No internet/phone service` no es un faltante: indica que el cliente no tiene el servicio base. Limpieza: `TotalCharges` llega como texto con 11 vacíos (0,16%, todos con `tenure=0`); se tipifica a número y se imputa con **mediana de entrenamiento** (los apuntes la recomiendan para variables sesgadas). No se eliminan filas, no se imputa 0, no se tratan outliers (son valores reales) y no hay duplicados.

## AED — hallazgos principales (sobre desarrollo)

- **Desbalance moderado:** 26,5% de bajas → medir con F1 y recall, no solo accuracy.
- **Perfil numérico:** quien se da de baja tiene menos antigüedad (media 18 vs 38 meses) y mayor cargo mensual (75 vs 61). `TotalCharges` es menor en bajas (clientes nuevos).
- **Antigüedad:** la tasa de baja cae de forma monótona: 47% (0–12 m) → 29% (13–24) → 20% (25–48) → 10% (49–72). Relación no lineal.
- **Asociación (Cramér's V):** Contract 0,41 · OnlineSecurity 0,35 · TechSupport 0,35 · InternetService 0,33 · PaymentMethod 0,31 son las más fuertes.
- **Sin asociación bivariada significativa (chi², p>0,05):** `gender` y `PhoneService`. No demuestra irrelevancia predictiva ni ausencia de interacciones; se conservan ambas.
- **Colinealidad:** `tenure`–`TotalCharges` r=0,83 → hay información compartida, no redundancia total demostrada. Dificulta interpretar coeficientes; en los árboles las variables correlacionadas pueden repartirse la importancia.
- **Perfil del churner:** contrato mes a mes + fibra óptica + cheque electrónico + poca antigüedad + sin seguridad/soporte + factura electrónica + adulto mayor.
- **FE exploratoria posterior** (cantidad_servicios, pago_automatico, Contract ordinal): se estudió después de observar el test original. Las comparaciones de ese test son descriptivas, no mejoras validadas ni fundamento para una nueva selección. Por ejemplo, pago_automatico mejora el test de RF, pero empeora su CV (0,7239 a 0,7183). El experimento estable y su selección original de logística quedan congelados; FE no se incorpora al pipeline y queda opcional en la app.

## Métricas: qué miden (coloquial + técnico)

Accuracy, precision, recall y F1 parten de la matriz de confusión al umbral fijado: TP (baja detectada), FN (baja que se escapó), FP (falsa alarma), TN (permanencia acertada). ROC-AUC y average precision usan las puntuaciones de riesgo en distintos umbrales, no una única matriz.

| Métrica | Técnica | En criollo |
|---|---|---|
| Accuracy | (TP+TN)/total | de cada 100, en cuántos acerté; puede ocultar fallos en las bajas |
| Precision (churn) | TP/(TP+FP) | cuando digo "se va", ¿cuántas veces acierto? → falsas alarmas |
| Recall (churn) | TP/(TP+FN) | de los que se van, ¿a cuántos agarré? → cobertura |
| F1 (churn) | 2·P·R/(P+R) | una nota que exige tener bien precision y recall a la vez |
| F1 macro | promedio del F1 de ambas clases | misma importancia a los que se van y a los que se quedan; **métrica de selección** |
| ROC-AUC | área TPR–FPR | capacidad de **ordenar** por riesgo sin depender del umbral (0,5 azar, 1 perfecto) |
| Average precision | precision ponderada por incrementos de recall | resume la curva precision-recall sin área trapezoidal; ranking enfocado en las bajas |

## Decisiones y preguntas probables

**¿Por qué clasificación y no regresión?** La salida es una de dos clases. Regresión logística, pese al nombre, es un clasificador probabilístico.

**¿Podría ser otro tipo de problema?** Según la Clase 2, es aprendizaje supervisado (hay etiqueta `Churn`) y clasificación (objetivo discreto, sin continuidad). No es regresión (haría falta un objetivo continuo, otro target); no es multiclase (solo dos clases); no es no supervisado —clustering o detección de anomalías ignorarían la etiqueta y responderían otra pregunta, y además la baja no es un evento raro—; no es por refuerzos (no hay agente ni recompensas). El alternativo más fiel sería análisis de supervivencia (tiempo-al-evento), descartado por no haber fecha ni horizonte; por eso hablamos de clasificación retrospectiva.

**¿Qué es leakage?** Que información ajena al entrenamiento influya en su ajuste o selección. Por eso split primero, EDA sobre desarrollo, pipeline dentro de CV y test después de fijar el ganador. El ID no predice; Churn nunca es entrada. Los controles del archivo completo no calculan transformaciones aprendidas.

**¿Por qué mediana y no cero?** Vacío significa desconocido. Que los 11 casos tengan antigüedad cero no demuestra su importe. La mediana usa solo entrenamiento y evita inventar una regla de negocio.

**¿Por qué one-hot y escala?** Las categorías no tienen distancias numéricas naturales. Logística y MLP son sensibles a magnitudes; StandardScaler usa media y desviación de entrenamiento. No fuerza normalidad. Los árboles no necesitan escala.

**¿Qué aporta la validación cruzada?** Estimar generalización durante selección usando varias particiones del desarrollo. No actualiza pesos por sí misma. El estimador sí ajusta parámetros en el entrenamiento de cada fold. El test queda separado y su score no decide hiperparámetros.

**¿Por qué F1 macro?** Da el mismo peso a cada clase y evita que domine la mayoría. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R). Accuracy=(TP+TN)/total. F1 macro promedia el F1 de ambas clases; F1 de churn mira solo la positiva. El dummy muestra qué pasa al ignorar bajas.

**¿Por qué umbral 0.5?** Es una decisión simple fijada antes de evaluar. Bajar el umbral suele aumentar recall y falsas alarmas. No lo modificamos con test; para optimizarlo hacen falta costos y validación sobre desarrollo.

**¿Cómo funciona Random Forest?** Entrena varios árboles sobre muestras bootstrap y aleatoriza características en sus divisiones. Combina probabilidades, no selecciona un único árbol ganador. Más árboles no eliminan automáticamente el sobreajuste.

**¿Cómo se entrena una MLP?** Forward calcula activaciones y probabilidad; entropía cruzada mide error; backprop aplica regla de la cadena para calcular gradientes; Adam actualiza pesos y sesgos. La fórmula de descenso básico es w ← w − η·gradiente; Adam adapta ese paso con momentos. Una época recorre entrenamiento; un batch es un subconjunto utilizado para una actualización.

**¿Arquitecturas e hiperparámetros?** 32 frente a 32→16 neuronas ocultas ReLU; salida sigmoide; Adam; tasa 0.001; batches 64; hasta 300 épocas; L2 alpha 0.001/0.01. Parada por pérdida de entrenamiento sin mejora suficiente durante más de 20 épocas, no por una validación interna. Ver épocas reales en la notebook. La comparación cambia profundidad y cantidad de parámetros.

**¿Convergieron las redes?** Ambas llegaron a 300 épocas y emitieron advertencia de no convergencia. Informamos los resultados con ese límite y no retocamos configuraciones a partir del test. Más épocas no garantizan mejorar validación; la brecha train–CV indica sobreajuste.

**¿Cómo miden el sobreajuste (overfitting)?** Comparando entrenamiento contra datos no vistos (la CV): una **brecha** train − CV grande es un indicio de sobreajuste. Una brecha pequeña no basta para afirmar buena generalización si ambos scores son bajos. En F1 macro: Logística 0,005 · Random Forest 0,044 · MLP 1 capa 0,090 · MLP 2 capas 0,173. En estas configuraciones, más capacidad amplía la brecha sin mejorar la CV. Logística se eligió por el mayor F1 macro medio de CV, antes de observar el test. Las curvas mostradas contienen únicamente pérdida de entrenamiento: no hay una curva de validación por época y su descenso, por sí solo, no demuestra sobreajuste ni generalización.

**¿La curva que baja prueba buen rendimiento?** No. Es pérdida de entrenamiento; se puede sobreajustar. Miramos brechas train-CV y prueba independiente. La pérdida no es F1 y no mide directamente errores de clasificación al umbral 0.5.

**¿Por qué recomendar ese modelo si otro tiene mejor test?** La regla fue el mayor F1 macro medio de CV. Elegir después según test contaminaría la selección. La prueba es una estimación final; diferencias pequeñas no acreditan superioridad estadística.

**¿Podemos decir que un contrato causa la baja o que evitaremos bajas?** No: son asociaciones en una muestra retrospectiva. No hay intervención ni horizonte temporal verificable. Un uso futuro necesita verificar disponibilidad de atributos antes del evento y validar en una cohorte posterior.

**¿Qué falta para uso real?** Definir horizonte y costos, revisar equidad/calibración y atributos disponibles, validar temporalmente y estudiar efectividad de intervenciones. Nada de eso se deduce de accuracy sola.

## Ensayo antes de entregar

Cada integrante debe explicar una matriz y un ejemplo de leakage, distinguir parámetro de hiperparámetro y describir forward/backprop/Adam sin leer. Revisar las tablas del informe contra la notebook, completar carátula y confirmar plazo/modalidad de WebCampus. No repetir como hechos las preguntas informales de los apuntes.
