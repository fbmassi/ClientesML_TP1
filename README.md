# TP1 · Machine Learning · Cliente

Trabajo académico de Inteligencia Artificial (UADE). Clasificación binaria de `Churn` con regresión logística, Random Forest y MLP de una y dos capas ocultas. La consigna se contrasta en la notebook; el informe es un borrador para completar y revisar por el grupo.

## Reproducir en Windows 11

Requisito: Python **3.13 de 64 bits** y acceso a los datos originales de la cátedra. Desde la raíz del repositorio, en PowerShell:

```powershell
cd TP1
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preparar_datos.py
.\.venv\Scripts\python.exe ejecutar_notebook.py
.\.venv\Scripts\python.exe -m unittest -v test_verificaciones
```

No es necesario activar el entorno ni cambiar la política de PowerShell. Ver [data/README.md](data/README.md) para ubicar el ZIP u obtener los originales. El CSV no se distribuye en Git. No se descarga un dataset alternativo.

Para abrir la notebook e interactuar (opcional):

```powershell
.\.venv\Scripts\python.exe -m notebook TP1_Cliente.ipynb
```

`ejecutar_notebook.py` elige explícitamente el Python que lo inicia y ejecuta todas las celdas. Si se ejecutan manualmente, seleccionar el kernel de `.venv`. Para entrenar sin interfaz ni volver a guardar la notebook:

```powershell
.\.venv\Scripts\python.exe experimento.py
```

Las dos formas entrenan de nuevo y reemplazan los resultados pequeños. No hace falta ejecutar ambas. Los tiempos dependen del equipo. Si se cambian código, versiones o datos, hay que volver a ejecutar la notebook, pruebas y actualizar las cifras del informe; los Markdown no se actualizan automáticamente.

## Método y archivos

Prueba estratificada 20%, semilla 42; CV estratificada de 3 folds en el 80% restante; dos configuraciones por modelo; selección por F1 macro de CV antes de test; umbral fijo 0.5. Preprocesamiento aprendido dentro del pipeline de cada fold. EDA sobre desarrollo. No se usan IDs como predictores.

| Archivo | Uso |
|---|---|
| `TP1_Cliente.ipynb` | Explicación académica, código, tablas y gráficos ejecutados |
| `experimento.py` | Funciones compartidas y entrenamiento por consola |
| `preparar_datos.py` | Extraer únicamente el CSV original de Cliente |
| `ejecutar_notebook.py` | Ejecución integral con el entorno correcto |
| `test_verificaciones.py` | Datos, particiones, transformaciones, métricas y notebook |
| `informe.md` | Borrador formal con resultados reales |
| `defensa_oral.md` | Decisiones, teoría, preguntas y exposición grupal |
| `resultados/` | Métricas, predicciones sin IDs, auditoría, versiones y cuatro figuras |

`resultados/validacion.csv` contiene los 8 candidatos y sus 3 scores individuales. `seleccion.csv` registra las configuraciones ganadoras y tiempos. `prueba.csv` incluye los cuatro modelos y el baseline. `experimento.json` contiene matrices de confusión, métricas por clase, curvas de pérdida y versiones; `auditoria.json` contiene controles e índices de la partición. No se guardan modelos ni caches entrenadas.

## Alcance y entrega

La etiqueta se clasifica retrospectivamente: faltan fecha de captura y horizonte de predicción para validar anticipación futura. La recomendación académica no equivale a validar un sistema de retención.

Antes de presentar: completar integrantes, legajos, comisión, docente vigente y fecha; revisar las decisiones entre todos; confirmar modalidad/plazo en WebCampus y subir allí código y soporte. El PDF original está fechado CII 2023: no se asume que esa sea la cursada actual. La asignación Cliente sigue la instrucción explícita del usuario; el PDF requiere respetar la asignación docente.

Los materiales originales, entorno, CSV y temporales están protegidos por `.gitignore`. Solo se trabaja en TP1, más ese ajuste general. Ver `resultados/verificacion.md` para la evidencia de esta ejecución.
