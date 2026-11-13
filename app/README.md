# App Streamlit — TP1 Churn de Clientes

App para **probar y comparar cada modelo** y ver el **AED** del dataset Cliente.
La lógica está separada por responsabilidad (igual criterio que el proyecto
`tpo-sueldos`): carga, limpieza, FE, procesamiento y un archivo por modelo.

## Estructura

```
app/
├── main.py                  # app Streamlit: AED en la página principal + selector de modelo
├── requirements.txt
└── src/
    ├── carga.py             # lee data/Cliente.csv (verifica SHA-256)
    ├── limpieza.py          # tipifica TotalCharges (espacios→NaN), mapea Churn
    ├── fe.py                # ingeniería de características opcional (3 derivadas)
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
- **FE opcional:** se puede activar la ingeniería de características evaluada
  (no mejora al modelo elegido; se deja para experimentar).

> Reproduce la metodología del informe. Nota: `data/Cliente.csv` no se versiona
> (ver `data/README.md`).
