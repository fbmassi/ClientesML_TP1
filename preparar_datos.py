"""Extrae exclusivamente el CSV original de Cliente, sin descargar sustitutos."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import zipfile

import py7zr

ROOT = Path(__file__).resolve().parent
DESTINO = ROOT / "data" / "Cliente.csv"


def preparar(zip_path):
    (ROOT / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cliente_", dir=ROOT / ".tmp") as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as archivo:
            candidatos = [n for n in archivo.namelist()
                          if "/Tema - Cliente/" in n and n.endswith("/Cliente.7z")]
            if len(candidatos) != 1:
                raise ValueError("Se esperaba exactamente un Tema - Cliente/Cliente.7z")
            siete = tmp / "Cliente.7z"
            siete.write_bytes(archivo.read(candidatos[0]))
        with py7zr.SevenZipFile(siete) as archivo:
            nombres = archivo.getnames()
            if any(Path(n).is_absolute() or ".." in Path(n).parts or ":" in n
                   for n in nombres):
                raise ValueError("Ruta insegura dentro de Cliente.7z")
            csvs = [n for n in nombres if n.lower().endswith(".csv")]
            if len(csvs) != 1:
                raise ValueError("Se esperaba exactamente un CSV de Cliente")
            archivo.extract(path=tmp / "cliente", targets=csvs)
        contenido = (tmp / "cliente" / csvs[0]).read_bytes()
        DESTINO.parent.mkdir(exist_ok=True)
        if DESTINO.exists() and DESTINO.read_bytes() != contenido:
            raise FileExistsError("Ya existe un Cliente.csv diferente; revisarlo antes de reemplazar")
        DESTINO.write_bytes(contenido)
        print(f"CSV original: {csvs[0]} -> {DESTINO}")
        print(f"SHA-256: {hashlib.sha256(contenido).hexdigest()}")
    return DESTINO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path,
                        default=ROOT / "_materiales_locales" / "TP 1 - Machine Learning.zip")
    preparar(parser.parse_args().zip)
