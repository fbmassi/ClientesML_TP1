"""Análisis exploratorio (AED), calculado sobre el set de desarrollo."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from .procesamiento import NUMERICAS_BASE, CATEGORICAS_BASE

COLORES = ["#287c8e", "#d97b31"]  # No / Yes


def resumen_target(dev):
    vc = dev["Churn"].value_counts().sort_index()
    pct = 100 * dev["Churn"].mean()
    return {"no": int(vc.get(0, 0)), "yes": int(vc.get(1, 0)), "pct_yes": pct}


def descriptivos_numericos(dev):
    return dev[NUMERICAS_BASE].describe().round(2)


def numericos_por_clase(dev):
    filas = []
    for c in NUMERICAS_BASE:
        g = dev.groupby("Churn")[c]
        filas.append({"variable": c,
                      "media_No": round(g.mean().get(0, np.nan), 2),
                      "media_Yes": round(g.mean().get(1, np.nan), 2),
                      "mediana_No": round(g.median().get(0, np.nan), 2),
                      "mediana_Yes": round(g.median().get(1, np.nan), 2)})
    return pd.DataFrame(filas)


def churn_por_tramo_tenure(dev):
    cortes = pd.cut(dev["tenure"], [-1, 12, 24, 48, 72],
                    labels=["0-12", "13-24", "25-48", "49-72"])
    t = dev.groupby(cortes, observed=True)["Churn"].agg(["size", "mean"])
    t["churn_%"] = (100 * t["mean"]).round(1)
    return t[["size", "churn_%"]].rename(columns={"size": "clientes"})


def _cramers_v(conf):
    chi2, p, _, _ = chi2_contingency(conf)
    n = conf.sum().sum()
    r, k = conf.shape
    return np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1)), p


def asociacion_categoricas(dev):
    filas = []
    for c in CATEGORICAS_BASE:
        v, p = _cramers_v(pd.crosstab(dev[c], dev["Churn"]).values)
        filas.append({"variable": c, "cramers_v": round(v, 3),
                      "significativa (p<0.05)": "sí" if p < 0.05 else "no"})
    return pd.DataFrame(filas).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def churn_por_categoria(dev, col):
    t = dev.groupby(col, observed=True)["Churn"].agg(["size", "mean"])
    t["churn_%"] = (100 * t["mean"]).round(1)
    return t[["size", "churn_%"]].rename(columns={"size": "clientes"}).sort_values("churn_%", ascending=False)


def correlaciones(dev):
    return dev[NUMERICAS_BASE].corr().round(3)


# ---- Figuras ----

def fig_target(dev):
    r = resumen_target(dev)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["No (0)", "Sí (1)"], [r["no"], r["yes"]], color=COLORES)
    ax.set_title(f"Distribución de churn ({r['pct_yes']:.1f}% se van)")
    ax.set_ylabel("Clientes")
    fig.tight_layout()
    return fig


def fig_numericas(dev):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, col in zip(axes, NUMERICAS_BASE):
        for clase, color in enumerate(COLORES):
            ax.hist(dev.loc[dev["Churn"] == clase, col].dropna(), bins=22,
                    alpha=.55, density=True, color=color, label=f"Churn={clase}")
        ax.set_title(col)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_categoria(dev, col):
    t = churn_por_categoria(dev, col)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(t.index.astype(str), t["churn_%"], color=COLORES[1])
    ax.set_title(f"% de baja por {col}")
    ax.set_xlabel("% churn")
    fig.tight_layout()
    return fig
