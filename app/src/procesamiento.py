"""Procesamiento: partición, preprocesador y entrenamiento/evaluación.

Reproduce la metodología del informe:
- Test 20% estratificado, semilla 42.
- StratifiedKFold k=3 sobre desarrollo; selección por F1-macro.
- Imputación por mediana + one-hot DENTRO del pipeline (sin leakage).
- Escalado solo para los modelos que lo requieren.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERICAS_BASE = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAS_BASE = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod"]


def construir_features(datos):
    """Devuelve (df, numericas, categoricas) con el set base de predictores."""
    return datos, list(NUMERICAS_BASE), list(CATEGORICAS_BASE)


def separar(df, predictoras, seed=SEED):
    return train_test_split(df[predictoras], df["Churn"], test_size=0.2,
                            random_state=seed, stratify=df["Churn"])


def construir_preprocesador(numericas, categoricas, escalar):
    pasos_num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    cat = Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                    ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", Pipeline(pasos_num), numericas),
                              ("cat", cat, categoricas)])


def _metricas(y, pred, prob):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_churn": precision_score(y, pred, zero_division=0),
        "recall_churn": recall_score(y, pred, zero_division=0),
        "f1_churn": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "average_precision": average_precision_score(y, prob),
    }


def entrenar_evaluar(datos, modulo_modelo, seed=SEED):
    """Entrena un modelo con CV, elige la mejor config y evalúa en test.

    Devuelve un dict con: métricas de prueba, matriz de confusión, curva ROC,
    resumen de CV, mejores hiperparámetros, brecha train–CV y avisos.
    """
    df, numericas, categoricas = construir_features(datos)
    predictoras = numericas + categoricas
    Xtr, Xte, ytr, yte = separar(df, predictoras, seed)

    spec = modulo_modelo.construir(seed)
    pipe = Pipeline([("preprocesar", construir_preprocesador(numericas, categoricas, spec["escalar"])),
                     ("modelo", spec["estimator"])])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scoring = {"f1_macro": "f1_macro"}

    with warnings.catch_warnings(record=True) as capturadas:
        warnings.simplefilter("always", ConvergenceWarning)
        busqueda = GridSearchCV(pipe, spec["grid"], scoring=scoring, refit=False,
                                cv=cv, n_jobs=1, return_train_score=True, error_score="raise")
        busqueda.fit(Xtr, ytr)
        res = pd.DataFrame(busqueda.cv_results_)
        mejor = int(res["mean_test_f1_macro"].to_numpy().argmax())
        params = res.iloc[mejor]["params"]
        modelo = clone(pipe).set_params(**params).fit(Xtr, ytr)
    avisos = sorted({str(w.message) for w in capturadas})

    prob = modelo.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    prob_d = dummy.predict_proba(Xte)[:, 1]
    pred_d = (prob_d >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(yte, prob)
    cv_f1 = float(res.iloc[mejor]["mean_test_f1_macro"])
    train_f1 = float(res.iloc[mejor]["mean_train_f1_macro"])
    return {
        "n_train": len(Xtr), "n_test": len(Xte),
        "n_features": modelo.named_steps["modelo"].n_features_in_,
        "mejores_params": params,
        "cv_f1_macro": cv_f1, "cv_f1_macro_std": float(res.iloc[mejor]["std_test_f1_macro"]),
        "train_f1_macro": train_f1, "brecha_overfitting": train_f1 - cv_f1,
        "metricas": _metricas(yte, pred, prob),
        "metricas_dummy": _metricas(yte, pred_d, prob_d),
        "matriz": confusion_matrix(yte, pred, labels=[0, 1]),
        "roc": (fpr, tpr),
        "avisos": avisos,
        "predictoras": predictoras,
    }
