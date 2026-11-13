"""Registro de modelos disponibles para la app (uno por archivo)."""
from . import logistica, random_forest, mlp_1capa, mlp_2capas

SEED = 42

# Orden de presentación en la app.
REGISTRO = {
    logistica.NOMBRE: logistica,
    random_forest.NOMBRE: random_forest,
    mlp_1capa.NOMBRE: mlp_1capa,
    mlp_2capas.NOMBRE: mlp_2capas,
}


def nombres():
    return list(REGISTRO.keys())


def obtener(nombre):
    return REGISTRO[nombre]
