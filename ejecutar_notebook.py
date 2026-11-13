"""Ejecuta y guarda la notebook con el mismo Python que inicia este comando."""
import os
from pathlib import Path
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def main():
    for variable, subdir in [("JUPYTER_RUNTIME_DIR", "runtime"), ("JUPYTER_CONFIG_DIR", "config"),
                              ("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "matplotlib")]:
        directorio = ROOT / ".jupyter" / subdir
        directorio.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directorio)
    path = ROOT / "TP1_Cliente.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # Evita seleccionar por accidente el Python global al ejecutar sin activar la venv.
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    cliente = NotebookClient(notebook, km=km, timeout=1200, startup_timeout=60,
                             resources={"metadata": {"path": str(ROOT)}})
    print("Ejecutando todas las celdas con", sys.executable, flush=True)
    try:
        cliente.execute()
    finally:
        # Con km explícito, el llamador es responsable del cierre del kernel.
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Notebook ejecutada sin errores:", sum(c.cell_type == "code" for c in notebook.cells), "celdas de código")


if __name__ == "__main__":
    main()
