"""
RUBINA - Revenue Management con Random Forest
=============================================
KPI: Net Yield = ingreso neto / cabina-noche disponible  (= ocupación x neto por cabina-noche vendida).
Ingreso neto de una reserva = ticket x (1 - comisión B2B 15 %) + gasto a bordo x 65 %.
  1. Random Forest (regresión): predice el gasto a bordo por cabina-noche de cada reserva.
     Se compara con regresión lineal y árbol de decisión; se seleccionan variables por importancia.
  2. Tarifa óptima: con el gasto a bordo predicho se prueban tarifas de 144 a 216 USD por temporada
     y se elige la que deja el mayor Net Yield. Supuesto: elasticidad 1.5 (se prueba de 0.3 a 2.0).
Ejecutar:  python rubina_revenue_rf.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "data" / "reporte_cruceros_revenue_management.csv"
SALIDAS = CARPETA / "salidas"

TEMPORADA = {1: "Baja", 2: "Baja", 11: "Baja", 3: "Media", 4: "Media", 5: "Media",
             9: "Media", 10: "Media", 6: "Alta", 7: "Alta", 8: "Alta", 12: "Alta"}
TARIFA_ACTUAL = {"Alta": 189.0, "Media": 180.0, "Baja": 165.6}  # USD por cabina-noche
COMISION_B2B = 0.15      # comisión que se lleva la agencia (solo reservas B2B)
MARGEN_BORDO = 0.65      # lo que queda del gasto a bordo después de su costo
ELASTICIDAD = 1.5        # si la tarifa baja 1 %, la demanda sube 1.5 % (supuesto)
VARIABLES = ["Barco", "Temporada", "Suite_Type", "Package", "Booking_Source", "Guest_Country",
             "Pasajeros_Reserva", "Cabinas_Reservadas", "Noches_Estancia", "Lead_Time_Dias"]


# ---------------------------------------------------------------- datos y KPI
def cargar_datos(archivo=DATOS):
    """Lee el CSV (ruta o archivo subido) y calcula el ingreso neto de cada reserva."""
    df = pd.read_csv(archivo, parse_dates=["Fecha_Viaje", "Fecha_Reserva"])
    df["Temporada"] = df["Fecha_Viaje"].dt.month.map(TEMPORADA)
    df["Gasto_Bordo"] = df["Gasto_Promedio_Diario_Huesped_USD"] * df["Pasajeros_Reserva"] * df["Noches_Estancia"]
    df["Ticket"] = df["Ingreso_Total_Reserva_USD"] - df["Gasto_Bordo"]
    comision = np.where(df["Booking_Source"] == "B2B", COMISION_B2B, 0)
    df["Ingreso_Neto"] = df["Ticket"] * (1 - comision) + df["Gasto_Bordo"] * MARGEN_BORDO
    df["Bordo_Cabina_Noche"] = df["Gasto_Promedio_Diario_Huesped_USD"] * df["Pasajeros_Reserva"] / df["Cabinas_Reservadas"]
    return df


def tabla_zarpes(df):
    """Un registro por zarpe: ocupación final y Net Yield."""
    z = df.groupby(["Barco", "Fecha_Viaje"]).agg(
        Temporada=("Temporada", "first"), Ruta=("Tipo_Ruta", "first"),
        Capacidad=("Cabinas_Totales_Barco", "first"), Noches=("Noches_Estancia", "first"),
        Ocupacion=("Porcentaje_Ocupacion_Ciclo", "first"),
        Neto=("Ingreso_Neto", "sum"), Cab=("Cabinas_Reservadas", "sum")).reset_index()
    z["Ocupacion"] = z["Ocupacion"] / 100
    z["Neto_por_cabina_noche"] = z["Neto"] / (z["Cab"] * z["Noches"])     # de las cabinas vendidas
    z["Net_Yield"] = z["Ocupacion"] * z["Neto_por_cabina_noche"]          # de las cabinas disponibles
    z["Cab_Noches_Disp"] = z["Capacidad"] * z["Noches"]
    return z.drop(columns=["Neto", "Cab"])


def net_yield(z):
    return float((z["Net_Yield"] * z["Cab_Noches_Disp"]).sum() / z["Cab_Noches_Disp"].sum())


# ---------------------------------------------------------------- 1. Random Forest
def entrenar(df, variables=VARIABLES, n_estimators=100, max_depth=10, criterion="squared_error",
             max_features=1.0, test_size=0.30, semilla=42):
    """Entrena el Random Forest y, con los mismos datos, una regresión lineal y un árbol de decisión."""
    X = pd.get_dummies(df[variables], drop_first=True).astype(float)
    y = df["Bordo_Cabina_Noche"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=semilla)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                               max_features=max_features, oob_score=True, n_jobs=-1, random_state=semilla)
    modelos = {"Regresión lineal": LinearRegression(),
               "Árbol de decisión": DecisionTreeRegressor(max_depth=max_depth, random_state=semilla),
               "Random Forest": rf}
    filas, pred = [], {}
    for nombre, m in modelos.items():
        m.fit(Xtr, ytr)
        pred[nombre] = m.predict(Xte)
        filas.append({"Modelo": nombre, "R2": r2_score(yte, pred[nombre]),
                      "RMSE": np.sqrt(mean_squared_error(yte, pred[nombre])),
                      "MAE": mean_absolute_error(yte, pred[nombre]),
                      "R2_OOB": rf.oob_score_ if m is rf else np.nan})
    return {"modelo": rf, "X": X, "X_prueba": Xte, "y_prueba": yte, "pred_prueba": pred["Random Forest"],
            "metricas": pd.DataFrame(filas), "variables": list(variables)}


def importancia(res, n=5000, semilla=42):
    """Importancia por permutación: cuánto empeora el R2 si se desordena cada variable."""
    m = res["X_prueba"].sample(min(n, len(res["X_prueba"])), random_state=semilla)
    p = permutation_importance(res["modelo"], m, res["y_prueba"].loc[m.index], n_repeats=3,
                               random_state=semilla, n_jobs=-1)
    imp = pd.Series(p.importances_mean, index=m.columns)
    imp = imp.groupby(lambda c: next(v for v in res["variables"] if c.startswith(v))).sum().clip(lower=0)
    return (imp / imp.sum()).sort_values(ascending=False).rename("Importancia")


def clasificacion(res):
    """Misma predicción vista como clase: ¿la reserva es de ALTO valor a bordo (sobre la mediana)?"""
    corte = res["y_prueba"].median()
    real, score = (res["y_prueba"] > corte).astype(int), res["pred_prueba"]
    clase = (score > corte).astype(int)
    fpr, tpr, _ = roc_curve(real, score)
    met = {"Accuracy": accuracy_score(real, clase), "Precision": precision_score(real, clase),
           "Recall": recall_score(real, clase), "F1": f1_score(real, clase), "AUC": roc_auc_score(real, score)}
    return met, confusion_matrix(real, clase), (fpr, tpr), corte


# ---------------------------------------------------------------- 2. tarifa que maximiza el profit
def valor_por_zarpe(df, res):
    """Por zarpe: % del ticket que se va en comisión y gasto a bordo PREDICHO por cabina-noche."""
    d = df.assign(Bordo_predicho=res["modelo"].predict(res["X"]),
                  cn=df["Cabinas_Reservadas"] * df["Noches_Estancia"],
                  com=np.where(df["Booking_Source"] == "B2B", COMISION_B2B, 0) * df["Ticket"])
    d["bordo_cn"] = d["Bordo_predicho"] * d["cn"]
    g = d.groupby(["Barco", "Fecha_Viaje"])[["com", "Ticket", "bordo_cn", "cn"]].sum()
    return pd.DataFrame({"Comision": g["com"] / g["Ticket"], "Bordo": g["bordo_cn"] / g["cn"]}).reset_index()


def curva_profit(z, temporada, elasticidad=ELASTICIDAD):
    """Prueba tarifas de 144 a 216 USD y recalcula cada zarpe histórico de la temporada."""
    zz = z[z["Temporada"] == temporada]
    peso = zz["Cab_Noches_Disp"] / zz["Cab_Noches_Disp"].sum()
    filas = []
    for tarifa in sorted(set(np.arange(144, 216.1, 1.0)) | {TARIFA_ACTUAL[temporada]}):   # incluye la actual
        cambio_demanda = (tarifa / TARIFA_ACTUAL[temporada]) ** (-elasticidad)
        ocupacion = np.minimum(zz["Ocupacion"] * cambio_demanda, 1.0)        # no se vende más que la capacidad
        neto = tarifa * (1 - zz["Comision"]) + zz["Bordo"] * MARGEN_BORDO    # profit de una cabina-noche vendida
        filas.append({"Temporada": temporada, "Tarifa": tarifa, "Ocupacion": (ocupacion * peso).sum(),
                      "Net_Yield": (ocupacion * neto * peso).sum()})
    return pd.DataFrame(filas)


def optimizar(z, valor, elasticidad=ELASTICIDAD):
    z = z.merge(valor, on=["Barco", "Fecha_Viaje"])
    curvas = pd.concat([curva_profit(z, t, elasticidad) for t in ["Alta", "Media", "Baja"]])
    filas = []
    for t, c in curvas.groupby("Temporada", sort=False):
        actual = c[c["Tarifa"] == TARIFA_ACTUAL[t]].iloc[0]
        mejor = c.loc[c["Net_Yield"].idxmax()]
        filas.append({"Temporada": t, "Tarifa_actual": TARIFA_ACTUAL[t], "Tarifa_optima": mejor["Tarifa"],
                      "Ocupacion_actual": actual["Ocupacion"], "Ocupacion_optima": mejor["Ocupacion"],
                      "NY_actual": actual["Net_Yield"], "NY_optimo": mejor["Net_Yield"],
                      "Peso": z.loc[z["Temporada"] == t, "Cab_Noches_Disp"].sum()})
    r = pd.DataFrame(filas)
    r["Mejora"] = r["NY_optimo"] / r["NY_actual"] - 1
    w = r["Peso"] / r["Peso"].sum()
    flota = {k: float((r[k] * w).sum()) for k in ["NY_actual", "NY_optimo", "Ocupacion_actual", "Ocupacion_optima"]}
    flota["Mejora"] = flota["NY_optimo"] / flota["NY_actual"] - 1
    return r.drop(columns="Peso"), curvas, flota


def sensibilidad(z, valor):
    """¿Desde qué elasticidad conviene bajar la tarifa? Se prueba de 0.3 a 2.0."""
    filas = []
    for e in np.round(np.arange(0.3, 2.001, 0.05), 2):
        tarifas, _, flota = optimizar(z, valor, e)
        filas.append({"Elasticidad": e, **flota,
                      **{f"Tarifa_{t}": v for t, v in zip(tarifas["Temporada"], tarifas["Tarifa_optima"])}})
    s = pd.DataFrame(filas)
    umbral = pd.Series({t: s.loc[s[f"Tarifa_{t}"] < TARIFA_ACTUAL[t], "Elasticidad"].min()
                        for t in ["Alta", "Media", "Baja"]}, name="Umbral")
    return s, umbral


if __name__ == "__main__":
    SALIDAS.mkdir(exist_ok=True)
    df = cargar_datos()
    z = tabla_zarpes(df)
    print(f"Net Yield actual: {net_yield(z):.1f} USD por cabina-noche disponible (meta +5 %: {net_yield(z) * 1.05:.1f})")

    # paso 1: bosque con las 10 variables -> importancia -> se quedan las que aportan >= 1 %
    res10 = entrenar(df)
    imp = importancia(res10)
    elegidas = list(imp[imp >= 0.01].index)
    res = entrenar(df, elegidas)                                          # paso 2: bosque final
    met_clase, matriz, _, corte = clasificacion(res)
    tabla = pd.concat([res10["metricas"].assign(Variables=10), res["metricas"].tail(1).assign(Variables=len(elegidas))])
    print("\nModelos (30 % de prueba):\n", tabla.round(3).to_string(index=False))
    print("\nImportancia:\n", imp.round(3).to_string(), "\nVariables elegidas:", elegidas)
    print(f"\nComo clasificador (alto valor a bordo > {corte:.0f} USD):", {k: round(v, 3) for k, v in met_clase.items()})
    print("Matriz de confusión [real x predicho]:\n", matriz)

    valor = valor_por_zarpe(df, res)
    tarifas, curvas, flota = optimizar(z, valor)
    sens, umbral = sensibilidad(z, valor)
    print("\nTarifa óptima:\n", tarifas.round(3).to_string(index=False))
    print("Flota:", {k: round(v, 4) for k, v in flota.items()})
    print("Bajar la tarifa conviene desde una elasticidad de:", umbral.to_dict())

    tabla.to_csv(SALIDAS / "rubina_modelos.csv", index=False)
    imp.to_csv(SALIDAS / "rubina_importancia.csv")
    pd.DataFrame([met_clase]).assign(Corte_USD=corte, VN=matriz[0, 0], FP=matriz[0, 1], FN=matriz[1, 0],
                                     VP=matriz[1, 1]).to_csv(SALIDAS / "rubina_clasificacion.csv", index=False)
    pd.DataFrame({"Real": res["y_prueba"].values, "Predicho": res["pred_prueba"]}).sample(
        3000, random_state=1).to_csv(SALIDAS / "rubina_real_vs_predicho.csv", index=False)
    tarifas.to_csv(SALIDAS / "rubina_tarifas.csv", index=False)
    curvas.to_csv(SALIDAS / "rubina_curvas.csv", index=False)
    sens.to_csv(SALIDAS / "rubina_sensibilidad.csv", index=False)
    umbral.rename_axis("Temporada").to_csv(SALIDAS / "rubina_umbral.csv")
