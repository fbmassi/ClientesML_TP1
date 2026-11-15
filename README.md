# TP1 de Inteligencia Artificial - Grupo 8

Clasificación de bajas de clientes con regresión logística, Random Forest y dos
arquitecturas MLP. La notebook es el soporte principal; Streamlit es opcional.
El informe definitivo está en [Word](entregables/TP1_Grupo8_Informe.docx) y
[PDF](entregables/TP1_Grupo8_Informe.pdf), dentro de `entregables/`.

## Windows 11 y PowerShell

Usar Python **3.13 de 64 bits** y abrir PowerShell en la carpeta que contiene
`experimento.py`. Crear un entorno limpio, sin copiar entornos de otra computadora:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

El dataset original **no está incluido**. Obtener de WebCampus de Inteligencia
Artificial el material del TP1 `TP 1 - Machine Learning.zip`, alternativa Cliente.
Reemplazar la ruta del ejemplo por la ubicación real del ZIP:

```powershell
.\.venv\Scripts\python.exe preparar_datos.py --zip "C:\ruta\TP 1 - Machine Learning.zip"
.\.venv\Scripts\python.exe ejecutar_notebook.py
.\.venv\Scripts\python.exe -m unittest -v test_verificaciones
.\.venv\Scripts\python.exe -m pip check
```

El preparador extrae únicamente Cliente y conserva sus bytes. Si el CSV original
ya está en `data/Cliente.csv`, omitir la extracción. Ver `data/README.md` para el
origen y la huella SHA-256. No usar un dataset sustituto. La reproducción requiere
acceso al material original de la cátedra; no se redistribuye con esta entrega.

Para abrir la notebook de forma interactiva:

```powershell
.\.venv\Scripts\python.exe -m notebook TP1_Cliente.ipynb
```

## Streamlit opcional

```powershell
.\.venv\Scripts\python.exe -m pip install -r app/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Estas dependencias incluyen las del TP y `py7zr`. No hace falta activar el entorno
ni cambiar la política de ejecución de PowerShell. La FE está desactivada por
defecto y es exploratoria: reutiliza un test ya observado y no cambia la selección
original. Ver `app/README.md`.

## Contenido y resultados

- `TP1_Cliente.ipynb`: explicación y ejecución completa.
- `experimento.py`: partición, pipelines, CV, evaluación y gráficos.
- `ejecutar_notebook.py` y `preparar_datos.py`: ejecución y preparación de datos.
- `test_verificaciones.py`: seis verificaciones de datos, aislamiento y resultados.
- `app/`: interfaz Streamlit y sus módulos.
- `resultados/`: métricas, predicciones, controles de partición y figuras.

Logística se selecciona por F1 macro medio de CV: **0.7325 ± 0.0115**; en prueba,
F1 macro **0.7376**. Las MLP alcanzaron 300 épocas con advertencias de no
convergencia, documentadas como limitación. No se ajustaron con el test.

## macOS y Linux

Usar Python 3.13. Desde la misma carpeta, crear el entorno con
`python3 -m venv .venv` y reemplazar `.\.venv\Scripts\python.exe` por
`.venv/bin/python` en los comandos anteriores. Adaptar también la ruta del ZIP.

## Google Colab opcional

Abrir la notebook es posible, pero para ejecutarla hay que subir y descomprimir
todo el ZIP de código; no alcanza con subir el archivo .ipynb. Aportar también
el CSV original en `data/Cliente.csv` o el ZIP de la cátedra para prepararlo.

Usar un runtime con Python 3.13. Desde la carpeta extraída que contiene
`experimento.py`, instalar las versiones declaradas con
`%pip install -r requirements.txt`; usar `%cd` para ubicarse allí.
Si se actualizan librerías ya cargadas, reiniciar la sesión y volver a esa carpeta
antes de ejecutar la notebook completa. No requiere Streamlit ni GPU.

Google documenta la [selección de versiones del runtime](https://research.google.com/colaboratory/runtime-version-faq.html).
Colab no se verificó mediante una ejecución real; Windows 11 sigue siendo el
entorno de referencia. No se garantizan métricas idénticas entre plataformas.
