# Datos originales de Cliente

Este directorio debe contener `Cliente.csv`. El CSV se excluye de Git y debe obtenerse del material original de la cátedra, como se indica abajo.

1. Obtener de WebCampus/cátedra el ZIP original `TP 1 - Machine Learning.zip`.
2. Ubicarlo en `_materiales_locales/`, dentro de la raíz del repositorio (también excluido de Git).
3. Desde la raíz de `ClientesML_TP1`, ejecutar `.\.venv\Scripts\python.exe preparar_datos.py` en PowerShell, después de instalar las dependencias.

Para otra ubicación, en PowerShell: `.\.venv\Scripts\python.exe preparar_datos.py --zip "C:\ruta\TP 1 - Machine Learning.zip"`.

El script lee exclusivamente `TP 1 - Machine Learning/TP 1 - Machine Learning/Tema - Cliente/Cliente.7z` y extrae su único CSV sin modificar sus bytes. No extrae Accidente ni las otras alternativas. No sobreescribe un CSV diferente ya existente.

- Original: `Cliente.csv`, 7.043 filas, 21 columnas.
- SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
- Separador: coma. Objetivo: `Churn` (`Yes`, `No`). Antigüedad real: `tenure`.
- `TotalCharges` incluye 11 campos con espacios; conservar el archivo original.

El script de análisis verifica la huella y falla si no coincide. El repositorio no concede permisos de redistribución de los materiales de la cátedra. Cada persona que reproduzca debe obtenerlos por su cuenta.
