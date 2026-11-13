"""MLP de una capa oculta (32 neuronas ReLU) — red neuronal."""
from sklearn.neural_network import MLPClassifier

NOMBRE = "MLP 1 capa"
ESCALAR = True

VENTAJAS = [
    "Capta relaciones no lineales con complejidad moderada.",
    "Arquitectura chica: menos riesgo de sobreajuste que una red profunda.",
]
LIMITACIONES = [
    "No convergió en 300 épocas (presupuesto fijo).",
    "Sensible a la escala y a la inicialización; poco interpretable.",
]


def construir(seed=42):
    base = dict(hidden_layer_sizes=(32,), activation="relu", solver="adam",
                learning_rate_init=0.001, batch_size=64, max_iter=300,
                early_stopping=False, n_iter_no_change=20, tol=1e-4,
                random_state=seed)
    return {
        "estimator": MLPClassifier(**base),
        "grid": {"modelo__alpha": [0.001, 0.01]},
        "escalar": ESCALAR,
    }
