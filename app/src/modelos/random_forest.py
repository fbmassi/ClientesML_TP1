"""Random Forest — conjunto de árboles con bootstrap y aleatoriedad de features."""
from sklearn.ensemble import RandomForestClassifier

NOMBRE = "Random Forest"
ESCALAR = False  # a un árbol no le importa la magnitud

VENTAJAS = [
    "Captura no linealidades e interacciones automáticamente.",
    "No requiere escalado; robusto frente a un árbol individual.",
    "Mejor ROC-AUC; permite ver importancia de variables.",
]
LIMITACIONES = [
    "Caja negra: menos interpretable que la logística.",
    "Más costoso; puede memorizar; las importancias no implican causalidad.",
    "Menor recall al umbral 0,5 en esta partición.",
]


def construir(seed=42):
    return {
        "estimator": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "grid": {"modelo__max_depth": [8, None]},
        "escalar": ESCALAR,
    }
