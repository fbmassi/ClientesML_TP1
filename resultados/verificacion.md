# Evidencia de verificación — 21/10/2026

## Ejecución de referencia

- Python 3.13 en `TP1/.venv`, Windows 11; versiones numéricas en `experimento.json` y `requirements.txt`.
- `python preparar_datos.py`: extracción exclusiva de Cliente; SHA-256 del CSV original verificado.
- `python ejecutar_notebook.py`: notebook ejecutada de principio a fin con kernel del entorno; **9 celdas de código, sin errores**.
- Entrenamiento real: 8 candidatos × 3 folds = 24 ajustes de validación, más 4 ajustes finales; baseline mayoritario adicional.
- Selección previa al test: Logistica, C=1.0, F1 macro CV 0.7325230048 ± 0.0115213106.
- Prueba independiente: accuracy 0.8055358410, F1 macro 0.7375828580, precision churn 0.6572327044, recall churn 0.5588235294, F1 churn 0.6040462428.
- Matriz del seleccionado: TN=926, FP=109, FN=165, TP=209.
- Ambas MLP llegaron a 300 épocas y emitieron advertencia de no convergencia. Se conservan y documentan estos resultados; no se cambiaron hiperparámetros después de consultar prueba.

## Verificaciones automáticas

Desde `TP1/` se ejecutó:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_verificaciones
.\.venv\Scripts\python.exe -m pip check
```

**6 pruebas aprobadas:**

1. CSV original, conversión de vacíos, etiquetas, exclusión de ID y objetivo.
2. Disjunción/cobertura de entrenamiento y prueba; folds que nunca incluyen prueba.
3. Imputación y escalado ajustados solo con entrenamiento; categorías desconocidas y valores extremos de prueba no cambian parámetros.
4. Métricas guardadas recalculadas desde las predicciones de los mismos clientes de prueba.
5. Ganador consistente con máximo de CV; scores individuales de folds, curvas y figuras presentes.
6. Notebook válida, ejecución secuencial de todas las celdas y ausencia de salidas de error.

`pip check`: sin dependencias incompatibles. Las dependencias fijadas se contrastaron con las instaladas. Las modificaciones posteriores al entrenamiento fueron de documentación y ubicación del cache de matplotlib; no cambiaron datos, algoritmos, configuraciones ni predicciones.

## Revisión de código, documentación y figuras

- Lectura íntegra de las dos páginas de la consigna y contraste con las tareas cubiertas en la notebook.
- Lectura de notebook/HTML originales y revisión de los pasajes relevantes de ML, ingeniería de características, redes y apuntes. `Clase 6.pdf` corresponde a otra materia.
- Revisión del flujo: split antes del EDA; pipelines completos dentro de CV; selección calculada antes de llamar a evaluación; umbral 0.5 fijo; no se reentrena según test.
- Las cuatro figuras se inspeccionaron visualmente: EDA, comparación, matrices y pérdida; títulos, ejes y leyendas legibles.
- Informe y guía oral contrastados contra las métricas, conteos y matrices guardadas.
- No se serializan modelos ni se incluyen IDs en las predicciones exportadas. Los índices son números de fila del CSV original.
- Git excluye materiales originales, CSV, entorno, caches y temporales. Se conservan los materiales preexistentes.

## Límites de lo verificado

No se afirma convergencia de las MLP, superioridad estadística universal, ausencia de sesgo, calibración operacional ni capacidad de anticipación temporal. No se optimizó el umbral con prueba ni se agregaron experimentos después de observarla. La prueba automática de aislamiento complementa la revisión del pipeline; por sí sola no demuestra ausencia de toda posible filtración semántica. Las fechas de captura y el horizonte no están disponibles.

La instalación y ejecución se verificaron en el entorno local preparado; no se probó una segunda máquina. El informe es un borrador: completar carátula, revisar en grupo y confirmar asignación/plazo/modalidad docente antes de entregar en WebCampus.
