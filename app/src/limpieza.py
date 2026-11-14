"""Limpieza de datos del dataset Cliente.

Decisiones (fundadas en los apuntes de Ciencia de Datos, Clase 3):
- `TotalCharges` viene como texto con 11 celdas en blanco (0,16%, todas con tenure=0):
  se convierte a número; los espacios pasan a NaN (ausencia != importe cero).
- La imputación por mediana NO se hace acá, sino dentro del pipeline por fold
  (procesamiento.py), para evitar fuga de información (leakage).
- `Churn` se mapea a 1/0.
- No hay duplicados ni inconsistencias; no se recortan extremos sin evidencia de error.
"""
import numpy as np
import pandas as pd


def limpiar(raw: pd.DataFrame) -> pd.DataFrame:
    """Tipifica y mapea el objetivo. No imputa (eso va en el pipeline)."""
    datos = raw.copy()
    # Espacios vacíos -> ausencia explícita; nunca un cero supuesto.
    datos["TotalCharges"] = pd.to_numeric(
        raw["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce",
    )
    datos["Churn"] = raw["Churn"].map({"No": 0, "Yes": 1})
    return datos


def resumen_limpieza(raw: pd.DataFrame, datos: pd.DataFrame) -> pd.DataFrame:
    """Tabla con los controles de calidad para mostrar en la app."""
    vacios_tc = int(raw["TotalCharges"].astype(str).str.strip().eq("").sum())
    filas = [
        ("Filas", len(datos), ""),
        ("Duplicados de fila", int(raw.duplicated().sum()), "0 esperado"),
        ("customerID duplicados", int(raw["customerID"].duplicated().sum()), "0 esperado"),
        ("TotalCharges vacíos", vacios_tc, "se imputan con mediana (en train)"),
        ("TotalCharges vacíos con tenure=0",
         int((datos["TotalCharges"].isna() & raw["tenure"].eq(0)).sum()),
         "coinciden todos"),
        ("Nulos tras tipificar (resto)",
         int(datos.drop(columns=["TotalCharges"]).isna().sum().sum()), "0 esperado"),
    ]
    return pd.DataFrame(filas, columns=["Control", "Valor", "Nota"])
