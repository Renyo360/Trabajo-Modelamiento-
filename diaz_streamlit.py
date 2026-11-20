"""
DÍAZ - Simulador del Net Yield (Streamlit)
==========================================
Una página para mover los parámetros de los tres modelos y ver su efecto en el KPI:
  Datos y KPI      ->  diagnóstico del problema y reducción de dimensión (PCA)
  Romero           ->  Holt-Winters, ARIMA y SARIMA (pronóstico de ocupación)
  Rubina           ->  Random Forest (gasto a bordo) + tarifa que maximiza el Net Yield
  Bustamante       ->  simulación de reservas: escenarios A, B y C
Ejecutar:  streamlit run diaz_streamlit.py
"""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

import bustamante_simulacion as bus
import romero_series_tiempo as rom
import rubina_revenue_rf as rub

ORD = ["Alta", "Media", "Baja"]
COL_T = {"Alta": "#2a78d6", "Media": "#eb6834", "Baja": "#1baf7a"}
COL_E = {"A": "#a8a8a3", "B": "#6da7ec", "C": "#1c5cab"}
COL_M = {"Holt-Winters": "#2a78d6", "ARIMA": "#eb6834", "SARIMA": "#1baf7a"}
NUMERICAS = ["Lead_Time_Dias", "Noches_Estancia", "Cabinas_Totales_Barco", "Cabinas_Reservadas",
             "Porcentaje_Ocupacion_Ciclo", "Pasajeros_Reserva", "Tripulacion", "Ingreso_Total_Reserva_USD",
             "Gasto_Promedio_Diario_Huesped_USD", "Puntuacion_Satisfaccion"]
ELEGIDAS = ["Pasajeros_Reserva", "Barco", "Cabinas_Reservadas", "Temporada"]   # resultado de la selección de Rubina

st.set_page_config(page_title="Net Yield · Cruceros", page_icon="🚢", layout="wide")
st.title("🚢 Simulador del Net Yield de una flota de cruceros")
st.caption("KPI: **Net Yield** = ingreso neto ÷ cabina-noche disponible. Hoy US$ 305.0 → meta US$ 320.2 (+5 %).")


# ---------------------------------------------------------------- funciones con caché
@st.cache_data(show_spinner="Leyendo el dataset...")
def leer(contenido):
    df = rub.cargar_datos(io.BytesIO(contenido))
    return df, rub.tabla_zarpes(df)


@st.cache_data(show_spinner="Ajustando Holt-Winters, ARIMA y SARIMA...")
def romero(contenido, meses_prueba, orden, orden_est, periodo, metodo, h):
    s = rom.serie_mensual(leer(contenido)[0])
    tabla, prueba = rom.evaluar(s, meses_prueba, orden=orden, orden_estacional=orden_est, periodo=periodo)
    metodo = tabla.loc[0, "Metodo"] if metodo == "El de menor MAPE" else metodo
    return s, tabla, prueba, rom.pronosticar(s, metodo, h, orden=orden, orden_estacional=orden_est, periodo=periodo), metodo


@st.cache_resource(show_spinner="Entrenando el Random Forest...")
def rubina_modelo(contenido, variables, n_estimators, max_depth, criterion, max_features, test_size):
    return rub.entrenar(leer(contenido)[0], list(variables), n_estimators, max_depth, criterion, max_features, test_size)


@st.cache_data(show_spinner="Calculando importancia de variables...")
def rubina_importancia(contenido, *param):
    return rub.importancia(rubina_modelo(contenido, *param), n=3000)


@st.cache_data(show_spinner="Buscando la tarifa óptima...")
def rubina_tarifas(contenido, elasticidad, *param):
    df, z = leer(contenido)
    valor = rub.valor_por_zarpe(df, rubina_modelo(contenido, *param))
    tarifas, curvas, flota = rub.optimizar(z, valor, elasticidad)
    sens, umbral = rub.sensibilidad(z, valor)
    return tarifas, curvas, flota, sens, umbral


@st.cache_data(show_spinner="Simulando reservas (30 réplicas tardan unos segundos)...")
def bustamante(contenido, tarifas, barco, conversion, b2b_c, dias, replicas, elasticidad):
    df = leer(contenido)[0]
    par, pesos = bus.parametros(df, dict(zip(ORD, tarifas)), barco, conversion, b2b_c, dias)
    return (*bus.correr(par, pesos, replicas, elasticidad), bus.tabla_arena(par, elasticidad), par)


def mostrar(t, formatos):
    """Tabla con números ya formateados ('—' si no aplica)."""
    t = t.copy()
    for c, f in formatos.items():
        t[c] = [("—" if pd.isna(v) else f(v)) if callable(f) else ("—" if pd.isna(v) else f.format(v)) for v in t[c]]
    st.dataframe(t, width="stretch", hide_index=True)


def figura(fig, alto=360):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- 1. datos
st.sidebar.header("📁 1. Carga de datos")
archivo = st.sidebar.file_uploader("Sube el CSV del dataset (Kaggle)", type=["csv"])
contenido = archivo.getvalue() if archivo else rub.DATOS.read_bytes()
st.sidebar.caption(f"Usando: **{archivo.name if archivo else 'dataset incluido en /data'}**")
try:
    df, z = leer(contenido)
except Exception as e:
    st.error(f"El archivo no tiene las columnas del dataset de cruceros ({e}).")
    st.stop()

st.sidebar.header("🧭 2. Sección")
seccion = st.sidebar.radio("Elige qué simular", ["Datos y KPI", "Romero · Series de tiempo",
                                                 "Rubina · Random Forest y tarifa", "Bustamante · Simulación"])
ny = rub.net_yield(z)

# ================================================================ DATOS Y KPI
if seccion == "Datos y KPI":
    ocup = (z["Ocupacion"] * z["Cab_Noches_Disp"]).sum() / z["Cab_Noches_Disp"].sum()
    c = st.columns(4)
    c[0].metric("Net Yield actual", f"US$ {ny:.1f}", f"meta US$ {ny * 1.05:.1f}", delta_color="off")
    c[1].metric("Ocupación", f"{ocup:.1%}", f"{1 - ocup:.1%} cabinas vacías", delta_color="inverse")
    c[2].metric("Gasto a bordo / ingreso", f"{df['Gasto_Bordo'].sum() / df['Ingreso_Total_Reserva_USD'].sum():.0%}")
    c[3].metric("Reservas · zarpes", f"{len(df):,} · {len(z):,}")
    st.dataframe(df.head(), width="stretch")

    a, b = st.columns(2)
    with a:
        figura(px.box(z.assign(Ocupacion=z["Ocupacion"] * 100), x="Temporada", y="Ocupacion", color="Temporada",
                      category_orders={"Temporada": ORD}, color_discrete_map=COL_T,
                      title="Ocupación por zarpe y temporada (%)").add_hline(y=100, line_dash="dash", line_color="#d03b3b"))
    with b:
        g = z.groupby("Temporada").apply(lambda x: np.average(x["Net_Yield"], weights=x["Cab_Noches_Disp"]),
                                         include_groups=False).reindex(ORD)
        figura(px.bar(x=g.index, y=g.values, color=g.index, color_discrete_map=COL_T, text_auto=".1f",
                      title="Net Yield por temporada (US$)", labels={"x": "", "y": "US$"}).add_hline(
            y=ny * 1.05, line_dash="dash", line_color="#d03b3b", annotation_text="meta flota"))
    m = z.assign(Mes=z["Fecha_Viaje"].dt.to_period("M").dt.to_timestamp()).query("Mes <= '2022-01-01'")
    m = m.groupby(["Mes", "Ruta"]).apply(lambda x: np.average(x["Ocupacion"], weights=x["Capacidad"]) * 100,
                                         include_groups=False).rename("Ocupacion").reset_index()
    figura(px.line(m, x="Mes", y="Ocupacion", color="Ruta", title="Ocupación mensual por tipo de ruta (%)"), 330)
    df["cn"] = df["Cabinas_Reservadas"] * df["Noches_Estancia"]
    comp = df.groupby("Temporada")[["Ticket", "Gasto_Bordo", "cn"]].sum().reindex(ORD)
    comp = pd.DataFrame({"Ticket": comp["Ticket"] / comp["cn"], "Gasto a bordo": comp["Gasto_Bordo"] / comp["cn"]})
    figura(px.bar(comp, barmode="stack", text_auto=".0f", title="Ingreso por cabina-noche vendida (US$)",
                  color_discrete_sequence=["#b7d3f6", "#2a78d6"], labels={"value": "US$", "variable": ""}), 330)

    st.subheader("Estadísticos descriptivos y reducción de dimensión (PCA)")
    st.dataframe(df[NUMERICAS].describe().T.round(2), width="stretch")
    k = st.slider("Número de componentes principales", 2, len(NUMERICAS), 4)
    pca = PCA(n_components=k).fit((df[NUMERICAS] - df[NUMERICAS].mean()) / df[NUMERICAS].std())
    var = pd.DataFrame({"Componente": [f"CP{i + 1}" for i in range(k)], "Varianza %": pca.explained_variance_ratio_ * 100})
    var["Acumulada %"] = var["Varianza %"].cumsum()
    a, b = st.columns([3, 2])
    with a:
        figura(px.bar(var, x="Componente", y="Varianza %", text_auto=".1f",
                      title=f"{k} componentes explican el {var['Acumulada %'].iloc[-1]:.1f} % de la varianza"), 320)
    with b:
        st.dataframe(pd.DataFrame(pca.components_.T, index=NUMERICAS, columns=var["Componente"]).round(2), width="stretch")

# ================================================================ ROMERO
elif seccion == "Romero · Series de tiempo":
    st.sidebar.header("⚙️ 3. Parámetros")
    periodo = st.sidebar.number_input("Período estacional (12 = mensual)", 2, 24, 12)
    tipo = st.sidebar.radio("Tipo de descomposición", ["additive", "multiplicative"], horizontal=True)
    meses_prueba = st.sidebar.slider("Meses de prueba", 6, 18, 12)
    st.sidebar.markdown("**ARIMA (p, d, q)**")
    p, d, q = (st.sidebar.number_input(n, 0, 5 if n != "d" else 2, 1) for n in ["p", "d", "q"])
    st.sidebar.markdown("**SARIMA estacional (P, D, Q)**")
    P, D, Q = (st.sidebar.number_input(n, 0, 2, 1) for n in ["P", "D", "Q"])
    h = st.sidebar.slider("Meses a pronosticar", 6, 24, 12)
    metodo = st.sidebar.selectbox("Método para el pronóstico", ["El de menor MAPE", "Holt-Winters", "ARIMA", "SARIMA"])

    st.header("Romero · ¿Cuánta demanda habrá cada mes?")
    st.write("Serie: **ocupación mensual de la flota** (cabinas reservadas ÷ disponibles). "
             "Los meses de demanda *Baja* son los que tienen más cabinas vacías y donde la tarifa debe ajustarse.")
    try:
        s, tabla, prueba, pron, usado = romero(contenido, meses_prueba, (p, d, q), (P, D, Q), periodo, metodo, h)
    except Exception as e:
        st.error(f"Con estos parámetros el modelo no se puede estimar: {e}")
        st.stop()

    st.subheader("1 · Identificación: descomposición de la serie")
    dec = rom.descomponer(s, periodo, tipo)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=["Serie", "Tendencia", "Estacionalidad", "Residuo"])
    for i, comp in enumerate([dec.observed, dec.trend, dec.seasonal, dec.resid], start=1):
        fig.add_trace(go.Scatter(x=comp.index, y=comp, mode="lines", line_color="#1c5cab", showlegend=False), row=i, col=1)
    figura(fig, 560)

    st.subheader(f"2-3 · Estimación y diagnóstico (últimos {meses_prueba} meses como prueba)")
    st.dataframe(tabla.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE_%": "{:.2f}", "LjungBox_p": "{:.3f}"})
                 .highlight_min(subset=["MAPE_%"], color="#d7e8fb"), width="stretch", hide_index=True)
    st.caption("MAPE ≤ 5 % = pronóstico útil. Ljung-Box p > 0.05 = los residuos no tienen patrón (modelo adecuado).")
    fig = go.Figure(go.Scatter(x=s.index, y=s, name="Real", line_color="#1a1a1a"))
    for mth in COL_M:
        fig.add_trace(go.Scatter(x=prueba.index, y=prueba[mth], name=mth, line=dict(color=COL_M[mth], dash="dash")))
    figura(fig.update_layout(title="Real vs. pronóstico en el período de prueba (%)"))

    st.subheader(f"4 · Pronóstico de {h} meses con {usado}")
    fig = go.Figure(go.Scatter(x=s.index, y=s, name="Histórico", line_color="#1a1a1a"))
    fig.add_trace(go.Scatter(x=pron["Mes"], y=pron["Pronostico"], name="Pronóstico", mode="lines+markers",
                             line_color="#1c5cab"))
    figura(fig)
    st.dataframe(pron.assign(Mes=pron["Mes"].dt.strftime("%Y-%m")).round(1), width="stretch", hide_index=True)

# ================================================================ RUBINA
elif seccion == "Rubina · Random Forest y tarifa":
    with st.sidebar.form("rf"):
        st.header("🧠 3. Random Forest")
        variables = st.multiselect("Variables predictoras (X)", rub.VARIABLES, default=ELEGIDAS)
        test_size = st.slider("Tamaño del set de prueba (%)", 10, 50, 30, step=5) / 100
        n_estimators = st.slider("Número de árboles (n_estimators)", 10, 500, 100, step=10)
        max_depth = st.slider("Máxima profundidad (max_depth)", 1, 50, 10)
        criterion = st.selectbox("Criterio (criterion)", ["squared_error", "friedman_mse", "poisson"])
        mf = st.selectbox("Variables por división (max_features)", ["1.0 (todas)", "sqrt", "0.5"])
        st.header("💲 4. Tarifa")
        elasticidad = st.slider("Elasticidad de la demanda (supuesto)", 0.3, 2.0, 1.5, step=0.05)
        st.form_submit_button("🚀 Entrenar y optimizar")
    if not variables:
        st.warning("Elige al menos una variable.")
        st.stop()
    param = (tuple(variables), n_estimators, max_depth, criterion,
             {"1.0 (todas)": 1.0, "sqrt": "sqrt", "0.5": 0.5}[mf], test_size)

    st.header("Rubina · ¿Qué tarifa deja más Net Yield?")
    st.write("El Random Forest predice el **gasto a bordo por cabina-noche** de cada reserva (60 % del ingreso). "
             "Con ese valor se prueba cada tarifa entre US$ 144 y 216 y se elige la que deja más Net Yield.")
    res = rubina_modelo(contenido, *param)
    t1, t2, t3 = st.tabs(["📊 Modelo (regresión)", "🎯 Vista de clasificación", "💲 Tarifa óptima"])
    with t1:
        rf = res["metricas"].set_index("Modelo").loc["Random Forest"]
        c = st.columns(4)
        c[0].metric("R² prueba", f"{rf['R2']:.3f}")
        c[1].metric("RMSE (US$)", f"{rf['RMSE']:.1f}")
        c[2].metric("MAE (US$)", f"{rf['MAE']:.1f}")
        c[3].metric("R² OOB", f"{rf['R2_OOB']:.3f}")
        mostrar(res["metricas"], {"R2": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}", "R2_OOB": "{:.3f}"})
        st.caption("Mismas variables y misma partición de prueba para los tres modelos (el árbol usa el mismo max_depth).")
        a, b = st.columns(2)
        with a:
            muestra = pd.DataFrame({"Real": res["y_prueba"].values, "Predicho": res["pred_prueba"]}).sample(
                min(3000, len(res["y_prueba"])), random_state=1)
            fig = px.scatter(muestra, x="Real", y="Predicho", opacity=0.35, title="Real vs. predicho (US$ por cabina-noche)")
            lim = [muestra.min().min(), muestra.max().max()]
            figura(fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#d03b3b", dash="dash"),
                                            showlegend=False)))
        with b:
            imp = rubina_importancia(contenido, *param) * 100
            figura(px.bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h", text_auto=".1f",
                          title="Importancia por permutación (%)", labels={"x": "%", "y": ""}).add_vline(
                x=1, line_dash="dash", line_color="#d03b3b"))
            st.caption("Selección de variables: se quedan las que aportan ≥ 1 %.")
    with t2:
        met, matriz, (fpr, tpr), corte = rub.clasificacion(res)
        st.write(f"La misma predicción, leída como clase: ¿la reserva es de **alto valor a bordo** "
                 f"(más de US$ {corte:.0f} por cabina-noche, la mediana)?")
        c = st.columns(5)
        for col, (k, v) in zip(c, met.items()):
            col.metric(k, f"{v:.3f}")
        a, b = st.columns(2)
        with a:
            figura(px.imshow(matriz, text_auto=True, color_continuous_scale="Blues", title="Matriz de confusión",
                             labels=dict(x="Predicho", y="Real"), x=["Bajo", "Alto"], y=["Bajo", "Alto"]))
        with b:
            fig = px.area(x=fpr, y=tpr, title=f"Curva ROC (AUC = {met['AUC']:.3f})",
                          labels={"x": "Tasa de falsos positivos", "y": "Tasa de verdaderos positivos"})
            figura(fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1))
    with t3:
        tarifas, curvas, flota, sens, umbral = rubina_tarifas(contenido, elasticidad, *param)
        st.session_state["tarifas_rm"] = tarifas.set_index("Temporada")["Tarifa_optima"].reindex(ORD).tolist()
        st.session_state["elasticidad"] = elasticidad
        c = st.columns(3)
        c[0].metric("Net Yield con tarifa actual", f"US$ {flota['NY_actual']:.1f}")
        c[1].metric("Net Yield con tarifa óptima", f"US$ {flota['NY_optimo']:.1f}", f"{flota['Mejora']:+.1%}")
        c[2].metric("Ocupación", f"{flota['Ocupacion_optima']:.1%}", f"{(flota['Ocupacion_optima'] - flota['Ocupacion_actual']) * 100:+.1f} pp")
        st.dataframe(tarifas.style.format({"Tarifa_actual": "{:.1f}", "Tarifa_optima": "{:.0f}", "Ocupacion_actual": "{:.1%}",
                                           "Ocupacion_optima": "{:.1%}", "NY_actual": "{:.1f}", "NY_optimo": "{:.1f}",
                                           "Mejora": "{:+.1%}"}), width="stretch", hide_index=True)
        a, b = st.columns(2)
        with a:
            fig = px.line(curvas, x="Tarifa", y="Net_Yield", color="Temporada", color_discrete_map=COL_T,
                          title="Net Yield según la tarifa (★ óptima)")
            fig.add_trace(go.Scatter(x=tarifas["Tarifa_optima"], y=tarifas["NY_optimo"], mode="markers", showlegend=False,
                                     marker=dict(symbol="star", size=14, color=[COL_T[t] for t in tarifas["Temporada"]])))
            figura(fig)
        with b:
            fig = px.line(sens.rename(columns={f"Tarifa_{t}": t for t in ORD}).melt("Elasticidad", ORD, "Temporada", "Tarifa"),
                          x="Elasticidad", y="Tarifa", color="Temporada", title="Tarifa óptima según la elasticidad",
                          color_discrete_map=COL_T)
            figura(fig.add_vline(x=elasticidad, line_dash="dash"))
        st.info(f"**Supuesto defendible:** bajar la tarifa conviene si la elasticidad supera "
                f"{umbral['Alta']:.2f} (Alta), {umbral['Media']:.2f} (Media) y {umbral['Baja']:.2f} (Baja). "
                "Por debajo, lo óptimo es mantener o subir la tarifa.")

# ================================================================ BUSTAMANTE
else:
    tar0 = st.session_state.get("tarifas_rm", [182.0, 167.0, 145.0])
    with st.sidebar.form("sim"):
        st.header("🎲 3. Simulación")
        barco = st.selectbox("Barco", sorted(df["Barco"].unique()), index=sorted(df["Barco"].unique()).index("Icon of the Seas"))
        st.markdown("**Tarifa escenarios B y C (US$)** · por defecto, la óptima de Rubina")
        tarifas = tuple(st.number_input(t, 100.0, 250.0, float(v), step=1.0) for t, v in zip(ORD, tar0))
        b2b_c = st.slider("Escenario C: % de reservas por agencia (B2B)", 10, 45, 30) / 100
        conversion = st.slider("% de solicitudes que compran a la tarifa actual", 40, 80, 60) / 100
        elasticidad = st.slider("Elasticidad", 0.3, 2.0, float(st.session_state.get("elasticidad", 1.5)), step=0.05)
        dias = st.slider("Días de venta por zarpe", 90, 270, 180, step=30)
        replicas = st.slider("Réplicas", 5, 50, 30, step=5)
        st.form_submit_button("🎲 Simular")

    st.header("Bustamante · ¿Qué pasa si cambiamos el proceso de reservas?")
    st.write("Cada solicitud llega (exponencial) → **¿hay cabinas?** → **¿acepta la tarifa?** → venta. "
             "**A** tarifa actual · **B** tarifa de Rubina · **C** tarifa de Rubina + campaña de venta directa.")
    res, por_temp, val, arena, par = bustamante(contenido, tarifas, barco, conversion, b2b_c, dias, replicas, elasticidad)
    r = res.set_index("Escenario")
    c = st.columns(3)
    for col, e, nombre in zip(c, "ABC", ["A · Actual", "B · Tarifa RM", "C · RM + venta directa"]):
        col.metric(nombre, f"US$ {r.loc[e, 'Net_Yield']:.1f}", None if e == "A" else f"{r.loc[e, 'Mejora_vs_A']:+.1%}")
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(x=list("ABC"), y=r["Net_Yield"], error_y=dict(type="data", array=r["IC95"]),
                               marker_color=[COL_E[e] for e in "ABC"], text=r["Net_Yield"].round(1)))
        fig.add_hline(y=r.loc["A", "Net_Yield"] * 1.05, line_dash="dash", line_color="#d03b3b", annotation_text="meta +5 %")
        figura(fig.update_layout(title="Net Yield anual simulado (US$, IC 95 %)"))
    with b:
        figura(px.bar(por_temp, x="Temporada", y="Net_Yield", color="Escenario", barmode="group", text_auto=".0f",
                      color_discrete_map=COL_E, category_orders={"Temporada": ORD}, title="Net Yield por temporada (US$)"))
    mostrar(res, {"Net_Yield": "{:.1f}", "Ocupacion": "{:.1%}", "IC95": "± {:.1f}", "Mejora_vs_A": "{:+.1%}",
                  "p_valor": lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"})
    st.caption("p-valor: prueba t pareada contra A (misma semilla en los tres escenarios). p < 0.05 = la mejora es significativa.")
    with st.expander("Validación: el escenario A reproduce lo histórico"):
        st.dataframe(val.round(3), width="stretch", hide_index=True)
    with st.expander("Parámetros para armar el mismo modelo en Arena"):
        st.markdown("**Create** (EXPO) → **Assign** (cabinas, huéspedes, gasto, comisión) → **Decide** ¿hay cabinas? → "
                    "**Decide** ¿acepta tarifa? → **Assign** + **Record** venta → **Dispose**. "
                    f"Réplica de {dias * 24:,} horas, {replicas} réplicas.")
        st.dataframe(arena, width="stretch", hide_index=True)
        st.download_button("Descargar tabla para Arena (CSV)", arena.to_csv(index=False).encode("utf-8-sig"),
                           "parametros_arena.csv", "text/csv")
