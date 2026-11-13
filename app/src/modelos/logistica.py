"""Regresión Logística — modelo lineal, interpretable. Baseline del TP."""
from sklearn.linear_model import LogisticRegression

NOMBRE = "Regresión Logística"
ESCALAR = True  # sensible a la escala de las variables

VENTAJAS = [
    "Interpretable: cada peso explica el efecto de su variable.",
    "Rápida de entrenar y devuelve probabilidades.",
    "Buen baseline; fue la de mayor F1-macro de CV.",
]
LIMITACIONES = [
    "Asume una frontera lineal en los predictores transformados.",
    "Coeficientes sensibles a la colinealidad (tenure–TotalCharges).",
]


def construir(seed=42):
    return {
        "estimator": LogisticRegression(max_iter=1000, random_state=seed),
        "grid": {"modelo__C": [0.1, 1.0]},
        "escalar": ESCALAR,
    }
