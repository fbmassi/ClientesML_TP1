"""Experimento reproducible de churn. La notebook explica y ejecuta estas funciones."""
import hashlib
import json
import os
import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# CLI y verificaciones también mantienen su cache dentro del proyecto.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".jupyter" / "matplotlib"))
(ROOT / ".jupyter" / "matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod"]
PREDICTORAS = NUMERICAS + CATEGORICAS


def cargar_datos(path=ROOT / "data" / "Cliente.csv"):
    """Validación del original y conversión determinista, sin estadísticas aprendidas."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Ejecutar primero: python preparar_datos.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("El CSV no coincide con el original asignado (SHA-256)")
    raw = pd.read_csv(path)
    if raw.shape != (7043, 21) or set(raw.columns) != set(PREDICTORAS + ["customerID", "Churn"]):
        raise ValueError("Esquema inesperado del dataset Cliente")
    if raw.customerID.isna().any() or raw.customerID.duplicated().any():
        raise ValueError("Se requiere un registro por cliente")
    if set(raw.Churn.unique()) != {"Yes", "No"}:
        raise ValueError("Etiquetas inesperadas")
    datos = raw.copy()
    # Los espacios vacíos son ausencia, nunca un importe cero supuesto.
    datos["TotalCharges"] = pd.to_numeric(raw.TotalCharges.str.strip().replace("", np.nan),
                                           errors="raise")
    datos["Churn"] = raw.Churn.map({"No": 0, "Yes": 1})
    return raw, datos


def separar(datos):
    """Split previo al EDA y al ajuste de cualquier transformación."""
    return train_test_split(datos[PREDICTORAS], datos.Churn, test_size=0.2,
                            random_state=SEED, stratify=datos.Churn)


def auditar(raw, datos, X_train, X_test, y_train, y_test):
    return {
        "sha256": SHA256, "filas": len(datos), "columnas": len(datos.columns),
        "nulos_originales": raw.isna().sum().astype(int).to_dict(),
        "vacios_originales": {c: int(raw[c].astype(str).str.strip().eq("").sum()) for c in raw},
        "nulos_convertidos": datos.isna().sum().astype(int).to_dict(),
        "duplicados_filas": int(raw.duplicated().sum()),
        "duplicados_id": int(raw.customerID.duplicated().sum()),
        "constantes_train": [c for c in X_train if X_train[c].nunique(dropna=False) <= 1],
        "clases_total": datos.Churn.value_counts().sort_index().astype(int).to_dict(),
        "clases_train": y_train.value_counts().sort_index().astype(int).to_dict(),
        "clases_test": y_test.value_counts().sort_index().astype(int).to_dict(),
        "n_train": len(X_train), "n_test": len(X_test),
        "indices_train": X_train.index.tolist(), "indices_test": X_test.index.tolist(),
        "faltantes_totalcharges_tenure_cero": bool(datos.loc[datos.TotalCharges.isna(), "tenure"].eq(0).all()),
        "faltantes_totalcharges_train": int(X_train.TotalCharges.isna().sum()),
        "faltantes_totalcharges_test": int(X_test.TotalCharges.isna().sum()),
    }


def describir_variables(raw, X_train):
    filas = []
    for c in raw:
        filas.append({"variable": c, "tipo_csv": str(raw[c].dtype),
                      "rol": "objetivo" if c == "Churn" else "identificador excluido" if c == "customerID" else "predictora",
                      "tipo_modelado": "numérica" if c in NUMERICAS else "categórica",
                      "valores_distintos_train": int(X_train[c].nunique()) if c in X_train else np.nan,
                      "faltantes_train": int(X_train[c].isna().sum()) if c in X_train else np.nan})
    return pd.DataFrame(filas)


def graficar_eda(X_train, y_train):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    colores = ["#287c8e", "#d97b31"]
    conteo = y_train.value_counts().sort_index()
    axes[0, 0].bar(["No (0)", "Sí (1)"], conteo, color=colores)
    axes[0, 0].set(title="Distribución de churn (entrenamiento)", ylabel="Clientes")
    for ax, col in zip(axes.flat[1:4], NUMERICAS):
        for clase, color in enumerate(colores):
            ax.hist(X_train.loc[y_train.eq(clase), col].dropna(), bins=22, alpha=.55,
                    density=True, color=color, label=f"Churn={clase}")
        ax.set(title=col, xlabel="Meses" if col == "tenure" else "Importe (unidad del CSV)", ylabel="Densidad")
        ax.legend()
    for ax, col in zip(axes.flat[4:], ["Contract", "InternetService"]):
        tabla = pd.crosstab(X_train[col], y_train, normalize="index")
        tabla[1].plot.barh(ax=ax, color=colores[1])
        ax.set(title=f"Proporción de bajas por {col}", xlabel="Proporción Churn=1", ylabel="", xlim=(0, 1))
    fig.suptitle("EDA del 80% de desarrollo: asociaciones descriptivas, sin interpretación causal", fontsize=14)
    return fig


def preprocesador(escalar=True):
    numericas = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        numericas.append(("escalar", StandardScaler()))
    categoricas = Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                             ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numericas", Pipeline(numericas), NUMERICAS),
                              ("categoricas", categoricas, CATEGORICAS)])


def candidatos():
    """Dos configuraciones por modelo: ocho candidatos y 24 ajustes de CV."""
    mlp = dict(activation="relu", solver="adam", learning_rate_init=.001,
               batch_size=64, max_iter=300, early_stopping=False,
               n_iter_no_change=20, tol=1e-4, random_state=SEED)
    especificaciones = {
        "Logistica": (LogisticRegression(max_iter=1000, random_state=SEED),
                       {"modelo__C": [.1, 1.0]}, True),
        "RandomForest": (RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                random_state=SEED, n_jobs=1),
                          {"modelo__max_depth": [8, None]}, False),
        "MLP_1_capa": (MLPClassifier(hidden_layer_sizes=(32,), **mlp),
                        {"modelo__alpha": [.001, .01]}, True),
        "MLP_2_capas": (MLPClassifier(hidden_layer_sizes=(32, 16), **mlp),
                         {"modelo__alpha": [.001, .01]}, True),
    }
    return {nombre: (Pipeline([("preprocesar", preprocesador(escalar)), ("modelo", modelo)]), grid)
            for nombre, (modelo, grid, escalar) in especificaciones.items()}


def entrenar(X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scoring = {"f1_macro": "f1_macro", "f1_churn": "f1", "recall_churn": "recall", "accuracy": "accuracy"}
    filas, resumen, modelos, avisos = [], [], {}, {}
    with threadpool_limits(limits=1):
        for nombre, (pipeline, grid) in candidatos().items():
            print(f"Validación: {nombre}", flush=True)
            # Sin refit automático: se registra por separado el tiempo del ajuste final.
            busqueda = GridSearchCV(pipeline, grid, scoring=scoring, refit=False,
                                    cv=cv, n_jobs=1, return_train_score=True, error_score="raise")
            inicio = time.perf_counter()
            with warnings.catch_warnings(record=True) as capturadas:
                warnings.simplefilter("always", ConvergenceWarning)
                busqueda.fit(X_train, y_train)
                resultados = pd.DataFrame(busqueda.cv_results_)
                mejor = int(resultados.mean_test_f1_macro.to_numpy().argmax())
                params = resultados.iloc[mejor]["params"]
                modelo = clone(pipeline).set_params(**params)
                inicio_refit = time.perf_counter()
                modelo.fit(X_train, y_train)
                refit_s = time.perf_counter() - inicio_refit
            avisos[nombre] = sorted({str(w.message) for w in capturadas})
            modelos[nombre] = modelo
            for i, r in resultados.iterrows():
                fila = {"modelo": nombre, "parametros": json.dumps(r["params"], sort_keys=True),
                        "seleccionado_en_familia": i == mejor,
                        "cv_f1_macro": r.mean_test_f1_macro, "cv_f1_macro_std": r.std_test_f1_macro,
                        "train_f1_macro": r.mean_train_f1_macro, "cv_f1_churn": r.mean_test_f1_churn,
                        "cv_recall_churn": r.mean_test_recall_churn, "cv_accuracy": r.mean_test_accuracy,
                        "fit_medio_s": r.mean_fit_time}
                for k in range(3):
                    fila[f"fold_{k+1}_f1_macro"] = r[f"split{k}_test_f1_macro"]
                filas.append(fila)
            r = resultados.iloc[mejor]
            resumen.append({"modelo": nombre, "parametros": json.dumps(params, sort_keys=True),
                            "cv_f1_macro": r.mean_test_f1_macro, "cv_f1_macro_std": r.std_test_f1_macro,
                            "train_f1_macro": r.mean_train_f1_macro,
                            "busqueda_y_refit_s": time.perf_counter() - inicio, "refit_s": refit_s})
    resumen = pd.DataFrame(resumen).sort_values("cv_f1_macro", ascending=False, kind="stable").reset_index(drop=True)
    # Regla fijada antes de test: máximo F1 macro medio. Empates exactos: orden simple primero.
    elegido = resumen.iloc[0].modelo
    return modelos, pd.DataFrame(filas), resumen, elegido, avisos


def metricas(y, pred, prob):
    return {"accuracy": accuracy_score(y, pred), "precision_churn": precision_score(y, pred, zero_division=0),
            "recall_churn": recall_score(y, pred, zero_division=0), "f1_churn": f1_score(y, pred, zero_division=0),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y, prob), "average_precision": average_precision_score(y, prob)}


def evaluar(modelos, X_train, y_train, X_test, y_test):
    """Se llama después de fijar el ganador; prueba común y umbral 0.5 fijo."""
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    filas, matrices, reportes = [], {}, {}
    predicciones = pd.DataFrame({"fila_original": X_test.index, "real": y_test.to_numpy()})
    for nombre, modelo in {"Dummy_mayoritaria": dummy, **modelos}.items():
        prob = modelo.predict_proba(X_test)[:, 1]
        pred = (prob >= .5).astype(int)
        filas.append({"modelo": nombre, **metricas(y_test, pred, prob)})
        matrices[nombre] = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
        reportes[nombre] = classification_report(y_test, pred, labels=[0, 1], target_names=["No", "Yes"],
                                               zero_division=0, output_dict=True)
        predicciones[f"{nombre}_prob"] = prob
        predicciones[f"{nombre}_pred"] = pred
    return pd.DataFrame(filas), matrices, reportes, predicciones


def graficar_evaluacion(resumen, prueba, matrices, modelos):
    fig1, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    nombres = resumen.modelo.tolist()
    x = np.arange(len(nombres))
    ax.bar(x-.2, resumen.cv_f1_macro, width=.4, yerr=resumen.cv_f1_macro_std, capsize=4, label="CV media ± DE (3 folds)")
    ax.bar(x+.2, prueba.set_index("modelo").loc[nombres, "f1_macro"], width=.4, label="Prueba independiente")
    ax.set(xticks=x, xticklabels=nombres, ylabel="F1 macro", ylim=(0, 1), title="Comparación: selección por CV, prueba como evaluación final")
    ax.legend(loc="lower right")
    fig2, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for ax, nombre in zip(axes.flat, modelos):
        ConfusionMatrixDisplay(np.array(matrices[nombre]), display_labels=["No", "Sí"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set(title=nombre, xlabel="Predicción", ylabel="Clase real")
    fig2.suptitle("Matrices de confusión en prueba: filas reales, columnas predichas")
    fig3, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for nombre, pipeline in modelos.items():
        modelo = pipeline.named_steps["modelo"]
        if hasattr(modelo, "loss_curve_"):
            ax.plot(np.arange(1, len(modelo.loss_curve_)+1), modelo.loss_curve_, label=nombre)
    ax.set(xlabel="Época (pasada por el entrenamiento)", ylabel="Pérdida de entrenamiento + penalización L2", title="Curvas reales de pérdida: ajustes finales de las MLP")
    ax.legend()
    return {"comparacion": fig1, "confusion": fig2, "perdida_mlp": fig3}


def guardar(raw, datos, X_train, X_test, y_train, y_test, modelos, cv, resumen, elegido, avisos,
            prueba, matrices, reportes, predicciones, eda, figuras, output=ROOT / "resultados"):
    output.mkdir(parents=True, exist_ok=True)
    auditoria = auditar(raw, datos, X_train, X_test, y_train, y_test)
    # Índices permiten comprobar disjunción/reproducibilidad sin publicar identificadores personales.
    (output / "auditoria.json").write_text(json.dumps(auditoria, indent=2, ensure_ascii=False), encoding="utf-8")
    cv.to_csv(output / "validacion.csv", index=False)
    resumen.to_csv(output / "seleccion.csv", index=False)
    prueba.to_csv(output / "prueba.csv", index=False)
    predicciones.to_csv(output / "predicciones_prueba.csv", index=False)
    describir_variables(raw, X_train).to_csv(output / "variables.csv", index=False)
    X_train[NUMERICAS].describe().to_csv(output / "descriptivos_train.csv")
    neuronales = {}
    for nombre, pipeline in modelos.items():
        m = pipeline.named_steps["modelo"]
        if isinstance(m, MLPClassifier):
            neuronales[nombre] = {"capas_ocultas": list(m.hidden_layer_sizes), "n_entradas": m.n_features_in_,
                "n_salidas": m.n_outputs_, "activacion_salida": m.out_activation_, "epocas": m.n_iter_,
                "alpha": m.alpha, "perdida_final": m.loss_, "curva_perdida": m.loss_curve_,
                "parametros_entrenables": int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))}
    metadatos = {"seleccionado_por_cv": elegido, "criterio": "max media F1 macro CV estratificada 3 folds",
                 "seed": SEED, "umbral": .5, "avisos": avisos, "redes": neuronales,
                 "matrices": matrices, "reportes_clasificacion": reportes,
                 "python": platform.python_version(), "plataforma": platform.platform(),
                 "versiones": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl", "nbformat", "nbclient", "ipykernel", "notebook", "py7zr"]}}
    (output / "experimento.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    for nombre, fig in {"eda": eda, **figuras}.items():
        fig.savefig(output / f"{nombre}.png", dpi=140)


def ejecutar():
    raw, datos = cargar_datos()
    X_train, X_test, y_train, y_test = separar(datos)
    eda = graficar_eda(X_train, y_train)
    modelos, cv, resumen, elegido, avisos = entrenar(X_train, y_train)
    prueba, matrices, reportes, predicciones = evaluar(modelos, X_train, y_train, X_test, y_test)
    figuras = graficar_evaluacion(resumen, prueba, matrices, modelos)
    guardar(raw, datos, X_train, X_test, y_train, y_test, modelos, cv, resumen, elegido, avisos,
            prueba, matrices, reportes, predicciones, eda, figuras)
    print(f"Seleccionado antes de prueba: {elegido}")
    print(prueba.round(4).to_string(index=False))
    plt.close("all")


if __name__ == "__main__":
    ejecutar()
