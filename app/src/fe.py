"""Ingeniería de características (opcional).

Se evaluaron tres transformaciones. En la comparación empírica NO mejoran al
modelo seleccionado (logística); se dejan como opción activable en la app.

- `cantidad_servicios` (0-7): conteo de servicios opcionales en 'Yes'
  (combinación por conocimiento de dominio).
- `pago_automatico` (0/1): binarización de PaymentMethod (automático vs manual).
- `Contract_ord` (0/1/2): codificación ordinal del plazo (mes<año<bienio).
"""
import pandas as pd

SERVICIOS_OPCIONALES = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
PAGOS_AUTOMATICOS = ["Bank transfer (automatic)", "Credit card (automatic)"]
ORDEN_CONTRATO = {"Month-to-month": 0, "One year": 1, "Two year": 2}


def agregar_fe(df: pd.DataFrame, cantidad_servicios=True, pago_automatico=True,
               contract_ordinal=True):
    """Devuelve (df_nuevo, numericas_agregadas, categoricas_a_quitar).

    `categoricas_a_quitar` indica qué columnas dejan de ir por one-hot
    (p. ej. Contract cuando se usa su versión ordinal).
    """
    out = df.copy()
    nuevas_num, quitar_cat = [], []
    if cantidad_servicios:
        out["cantidad_servicios"] = sum(
            (out[c] == "Yes").astype(int) for c in SERVICIOS_OPCIONALES)
        nuevas_num.append("cantidad_servicios")
    if pago_automatico:
        out["pago_automatico"] = out["PaymentMethod"].isin(PAGOS_AUTOMATICOS).astype(int)
        nuevas_num.append("pago_automatico")
    if contract_ordinal:
        out["Contract_ord"] = out["Contract"].map(ORDEN_CONTRATO)
        nuevas_num.append("Contract_ord")
        quitar_cat.append("Contract")
    return out, nuevas_num, quitar_cat
