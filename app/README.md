# App Streamlit — TP1 Churn de Clientes

Complemento opcional para mostrar el AED y reproducir cada uno de los cuatro modelos del experimento base. La notebook sigue siendo el soporte académico principal. La opción FE conserva los experimentos exploratorios posteriores del compañero, desactivada por defecto.

## Ejecutar en Windows 11

Usar Python **3.13 de 64 bits**, sin reutilizar un entorno copiado de otra computadora. Desde la raíz de `ClientesML_TP1` (la carpeta que contiene `experimento.py`):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r app/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Si ya se creó el entorno siguiendo el README principal, omitir la primera línea. `app/requirements.txt` incluye las dependencias principales del TP, con `py7zr`, y agrega la versión de Streamlit verificada. No hace falta activar el entorno ni cambiar la política de PowerShell.

El CSV debe estar en `data/Cliente.csv`. Si falta, obtener el ZIP original según `data/README.md` y ejecutar:

```powershell
.\.venv\Scripts\python.exe preparar_datos.py --zip "C:\ruta\TP 1 - Machine Learning.zip"
```

No hace falta volver a extraer si el CSV original ya está presente. No se distribuyen entornos virtuales ni los materiales completos de la cátedra.

## macOS / Linux

Desde la misma raíz, crear un entorno limpio e instalar las dependencias:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r app/requirements.txt
# Si falta el CSV, obtener el ZIP original y prepararlo antes de iniciar:
.venv/bin/python preparar_datos.py --zip "/ruta/TP 1 - Machine Learning.zip"
.venv/bin/python -m streamlit run app/main.py
```

## Qué hace

- **AED:** distribución del objetivo, numéricas por clase, tramos de antigüedad, correlaciones y asociación bivariada (Cramér's V / chi²), sobre el **80% de desarrollo**.
- **Selector de modelo:** reproduce la partición estratificada 80/20, semilla 42, CV de 3 folds y selección de configuración por F1 macro. Muestra métricas, baseline, matriz, ROC, brecha train–CV y advertencias. El modelo recomendado del experimento original sigue siendo logística, seleccionado originalmente por CV.
- **FE opcional:** los resultados se identifican como exploratorios porque reutilizan un test ya observado. No son una nueva evaluación independiente ni justifican cambiar el modelo, umbral o pipeline estable. No seleccionar variantes por las cifras de ese test.

Una asociación bivariada no significativa no demuestra irrelevancia predictiva. Una brecha train–CV grande sugiere sobreajuste; una pequeña debe interpretarse junto con el nivel de los scores. Las advertencias de no convergencia de las MLP se mantienen visibles.

Al cambiar el CSV local, la app vuelve a leer los datos; la caché de entrenamiento distingue su contenido y las opciones de FE.

## Estructura

`main.py` contiene la interfaz. `src/` conserva los módulos de carga, limpieza, ingeniería de características, procesamiento, AED y un archivo por modelo. Las correcciones documentales no cambian estimadores, hiperparámetros, particiones ni transformaciones aprendidas.
