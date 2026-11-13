"""MLP de dos capas ocultas (32 -> 16 ReLU) — red neuronal más profunda."""
from sklearn.neural_network import MLPClassifier

NOMBRE = "MLP 2 capas"
ESCALAR = True

VENTAJAS = [
    "Composiciones no lineales más ricas (mayor capacidad).",
    "Permite comparar el efecto de agregar profundidad.",
]
LIMITACIONES = [
    "Más parámetros -> mayor sobreajuste (brecha train–CV 0,17).",
    "No convergió en 300 épocas; más profundidad no mejoró el resultado.",
]


def construir(seed=42):
    base = dict(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                learning_rate_init=0.001, batch_size=64, max_iter=300,
                early_stopping=False, n_iter_no_change=20, tol=1e-4,
                random_state=seed)
    return {
        "estimator": MLPClassifier(**base),
        "grid": {"modelo__alpha": [0.001, 0.01]},
        "escalar": ESCALAR,
    }
