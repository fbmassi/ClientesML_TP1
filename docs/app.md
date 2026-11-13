# App Streamlit — TP1 Churn de Clientes

App para **probar y comparar cada modelo** y ver el **AED** del dataset Cliente.
La lógica está separada por responsabilidad (igual criterio que el proyecto
`tpo-sueldos`): carga, limpieza, procesamiento y un archivo por modelo.

## Estructura

```
app/
├── main.py                  # app Streamlit: AED en la página principal + selector de modelo
├── requirements.txt
└── src/
    ├── carga.py             # lee data/Cliente.csv (verifica SHA-256)
    ├── limpieza.py          # tipifica TotalCharges (espacios→NaN), mapea Churn
    ├── procesamiento.py     # split 80/20, preprocesador (ColumnTransformer), CV + evaluación
    ├── aed.py               # análisis exploratorio (estadísticos, chi²/Cramér's V, figuras)
    └── modelos/
        ├── logistica.py     # Regresión Logística
        ├── random_forest.py # Random Forest
        ├── mlp_1capa.py     # MLP 1 capa (32)
        └── mlp_2capas.py    # MLP 2 capas (32→16)
```

## Cómo correrla

Desde la carpeta `TP1/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r app/requirements.txt
.venv/bin/python preparar_datos.py          # deja data/Cliente.csv
.venv/bin/streamlit run app/main.py
```

## Qué hace

- **Página principal (AED):** desbalance del objetivo, distribución de las
  numéricas por clase, churn por tramo de antigüedad, correlaciones y la
  fuerza de asociación (Cramér's V / chi²) de cada categórica. Todo sobre el
  **80% de desarrollo** (no se mira el test).
- **Selector de modelo (barra lateral):** elegí entre los 4 modelos; se entrena
  con **StratifiedKFold k=3**, se elige la mejor configuración por **F1-macro**
  y se evalúa en el **test 20%**. Muestra métricas vs baseline, matriz de
  confusión, curva ROC, brecha de sobreajuste (train–CV) y ventajas/limitaciones.

> Reproduce la metodología del informe. La ingeniería de características evaluada
> (y descartada por no aportar al modelo elegido) se documenta en `informe.md` §3.

## Deploy (Streamlit Cloud)

El dataset de la cátedra **no se versiona**, así que en un deploy no está el
`Cliente.csv`. La app lo detecta y muestra un **cargador de archivos**: subí
`Cliente.csv` (el dataset Telco Churn) y la app arranca. En local, en cambio,
toma el archivo de `data/Cliente.csv` automáticamente. Nota: `data/Cliente.csv` no se versiona
> (ver `data.md`).
