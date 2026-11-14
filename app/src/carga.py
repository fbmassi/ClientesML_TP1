"""Carga del dataset Cliente. Solo lectura del CSV original, sin transformar."""
from pathlib import Path
import hashlib
import pandas as pd

# app/src/carga.py -> parents[2] = carpeta TP1
ROOT_TP1 = Path(__file__).resolve().parents[2]
DATA = ROOT_TP1 / "data" / "Cliente.csv"
SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"

COLUMNAS_ESPERADAS = 21
FILAS_ESPERADAS = 7043


def cargar_crudo(path: Path = DATA) -> pd.DataFrame:
    """Lee el CSV original tal cual (todas las columnas como vienen).

    Devuelve el DataFrame crudo. Verifica la huella SHA-256 del archivo
    asignado por la cátedra; si no coincide, avisa pero no detiene la app.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecutar antes 'python preparar_datos.py'."
        )
    huella = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = pd.read_csv(path)
    raw.attrs["sha256_ok"] = (huella == SHA256)
    raw.attrs["sha256"] = huella
    return raw


def resumen_carga(raw: pd.DataFrame) -> dict:
    """Metadatos básicos para mostrar en la app."""
    return {
        "filas": len(raw),
        "columnas": raw.shape[1],
        "sha256_ok": bool(raw.attrs.get("sha256_ok", False)),
        "esquema_ok": raw.shape == (FILAS_ESPERADAS, COLUMNAS_ESPERADAS),
    }
