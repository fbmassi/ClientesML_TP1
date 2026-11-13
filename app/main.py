"""TP1 — Machine Learning: clasificación de bajas de clientes (Cliente).

App Streamlit: AED en la página principal + selector de modelo para entrenar
y evaluar. Lógica separada en src/ (carga, limpieza, fe, procesamiento, aed,
modelos/<cada modelo>).

Ejecutar:  streamlit run app/main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src import carga, limpieza, aed, procesamiento
from src import modelos as registro_modelos

st.set_page_config(page_title="TP1 ML · Churn Cliente", layout="wide")


@st.cache_data(show_spinner=False)
def cargar():
    raw = carga.cargar_crudo()
    datos = limpieza.limpiar(raw)
    return raw, datos


@st.cache_data(show_spinner="Entrenando y evaluando…")
def entrenar(nombre_modelo, usar_fe, serv, pago, contr):
    _, datos = cargar()
    modulo = registro_modelos.obtener(nombre_modelo)
    return procesamiento.entrenar_evaluar(
        datos, modulo, usar_fe=usar_fe,
        cantidad_servicios=serv, pago_automatico=pago, contract_ordinal=contr)


# ---------------- Carátula ----------------
st.title("TP1 · Machine Learning — Churn de clientes")
st.caption("Inteligencia Artificial · 2C 2026 · Prof. Parkinson · "
           "Equipo 8: Massi (1156946), Nari (1165816), Vanore (1179980)")

raw, datos = cargar()
meta = carga.resumen_carga(raw)
c1, c2, c3 = st.columns(3)
c1.metric("Clientes", meta["filas"])
c2.metric("Columnas", meta["columnas"])
c3.metric("SHA-256 original", "OK" if meta["sha256_ok"] else "≠")

# Partición de desarrollo (para el AED, sin mirar el test)
df_base, numericas, categoricas = procesamiento.construir_features(datos, usar_fe=False)
Xtr, Xte, ytr, yte = procesamiento.separar(df_base, numericas + categoricas)
dev = Xtr.copy()
dev["Churn"] = ytr.values

# ---------------- Sidebar: modelo ----------------
st.sidebar.header("Modelo")
nombre_modelo = st.sidebar.selectbox("Elegí un modelo", registro_modelos.nombres())
st.sidebar.markdown("**Ingeniería de características (opcional)**")
usar_fe = st.sidebar.checkbox("Activar FE derivada", value=False,
                              help="Evaluada: no mejora al modelo elegido. Se deja opcional.")
serv = st.sidebar.checkbox("cantidad_servicios", value=True, disabled=not usar_fe)
pago = st.sidebar.checkbox("pago_automatico", value=True, disabled=not usar_fe)
contr = st.sidebar.checkbox("Contract ordinal", value=True, disabled=not usar_fe)
entrenar_click = st.sidebar.button("Entrenar y evaluar", type="primary")

# ---------------- Página principal: AED ----------------
tab_aed, tab_modelo = st.tabs(["📊 Análisis exploratorio (AED)", "🤖 Modelo"])

with tab_aed:
    st.subheader("Análisis exploratorio (sobre desarrollo, 80%)")
    r = aed.resumen_target(dev)
    st.write(f"**Desbalance:** {r['yes']} bajas / {r['no']} permanencias "
             f"(**{r['pct_yes']:.1f}%** se van). Por eso se mide con F1/recall, no solo accuracy.")
    st.pyplot(aed.fig_target(dev))

    st.markdown("**Numéricas: distribución por clase**")
    st.pyplot(aed.fig_numericas(dev))
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("Medias/medianas por clase")
        st.dataframe(aed.numericos_por_clase(dev), width='stretch')
        st.markdown("Churn por tramo de antigüedad")
        st.dataframe(aed.churn_por_tramo_tenure(dev), width='stretch')
    with cc2:
        st.markdown("Correlación entre numéricas")
        st.dataframe(aed.correlaciones(dev), width='stretch')
        st.caption("tenure–TotalCharges r≈0,83: TotalCharges es redundante.")

    st.markdown("**Asociación de cada categórica con la baja (Cramér's V, chi²)**")
    st.dataframe(aed.asociacion_categoricas(dev), width='stretch', height=320)
    st.caption("gender y PhoneService no son significativas (p>0,05).")

    col = st.selectbox("Ver % de baja por categoría",
                       procesamiento.CATEGORICAS_BASE, index=procesamiento.CATEGORICAS_BASE.index("Contract"))
    cols = st.columns([1, 1])
    cols[0].dataframe(aed.churn_por_categoria(dev, col), width='stretch')
    cols[1].pyplot(aed.fig_categoria(dev, col))

with tab_modelo:
    st.subheader(f"Modelo: {nombre_modelo}")
    modulo = registro_modelos.obtener(nombre_modelo)
    cvent, clim = st.columns(2)
    cvent.markdown("**Ventajas**\n\n" + "\n".join(f"- {v}" for v in modulo.VENTAJAS))
    clim.markdown("**Limitaciones**\n\n" + "\n".join(f"- {v}" for v in modulo.LIMITACIONES))

    if not entrenar_click:
        st.info("Elegí un modelo en la barra lateral y tocá **Entrenar y evaluar**.")
    else:
        res = entrenar(nombre_modelo, usar_fe, serv, pago, contr)
        m, d = res["metricas"], res["metricas_dummy"]
        st.markdown("**Métricas en prueba** (test 20%, umbral 0,5)")
        tabla = pd.DataFrame({
            nombre_modelo: m,
            "Baseline (Dummy)": d,
        }).T[["accuracy", "precision_churn", "recall_churn", "f1_churn", "f1_macro", "roc_auc"]]
        st.dataframe(tabla.round(4), width='stretch')

        k1, k2, k3 = st.columns(3)
        k1.metric("F1-macro (CV)", f"{res['cv_f1_macro']:.4f}", f"±{res['cv_f1_macro_std']:.4f}")
        k2.metric("F1-macro (prueba)", f"{m['f1_macro']:.4f}")
        k3.metric("Brecha train–CV (overfitting)", f"{res['brecha_overfitting']:.4f}")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Matriz de confusión** (filas: real, columnas: predicho)")
            mat = pd.DataFrame(res["matriz"], index=["Real No", "Real Sí"],
                               columns=["Pred No", "Pred Sí"])
            st.dataframe(mat, width='stretch')
        with g2:
            import matplotlib.pyplot as plt
            fpr, tpr = res["roc"]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(fpr, tpr, color="#287c8e", label=f"AUC={m['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Curva ROC"); ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

        st.caption(f"Mejores hiperparámetros: {res['mejores_params']} · "
                   f"{res['n_features']} features · FE: {'sí' if res['usar_fe'] else 'no'}")
        if res["avisos"]:
            st.warning("Avisos de entrenamiento: " + " | ".join(res["avisos"]))
