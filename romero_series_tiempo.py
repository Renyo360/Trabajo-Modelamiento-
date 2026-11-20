"""
ROMERO - Pronóstico de reservas con series de tiempo
===================================================
Serie: ocupación mensual de la flota = cabinas reservadas / cabinas disponibles (%).
Pasos (metodología Box-Jenkins vista en clase):
  1. Identificar: gráfico, descomposición aditiva (tendencia + estacionalidad + residuo) y prueba ADF.
  2. Estimar: Holt-Winters, ARIMA(p,d,q) y SARIMA(p,d,q)(P,D,Q,12).
  3. Diagnosticar: error en los últimos 12 meses (MAE, RMSE, MAPE) y Ljung-Box de los residuos.
  4. Pronosticar: el mejor método proyecta los próximos 12 meses.
Ejecutar:  python romero_series_tiempo.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
CARPETA = Path(__file__).resolve().parent


def serie_mensual(df):
    """Ocupación mensual de la flota (%) por mes de zarpe."""
    df = df.assign(Fecha_Viaje=pd.to_datetime(df["Fecha_Viaje"]))
    z = df.groupby(["Barco", "Fecha_Viaje"]).agg(Ocupacion=("Porcentaje_Ocupacion_Ciclo", "first"),
                                                 Capacidad=("Cabinas_Totales_Barco", "first")).reset_index()
    z["Reservadas"] = z["Ocupacion"] / 100 * z["Capacidad"]
    m = z.groupby(z["Fecha_Viaje"].dt.to_period("M"))[["Reservadas", "Capacidad"]].sum()
    s = (m["Reservadas"] / m["Capacidad"] * 100).rename("Ocupacion")
    s.index = s.index.to_timestamp()
    zarpes = z.groupby(z["Fecha_Viaje"].dt.to_period("M")).size().values
    return s[zarpes >= 0.5 * np.median(zarpes)].asfreq("MS")     # quita meses incompletos (feb-2022)


def descomponer(s, periodo=12, tipo="additive"):
    return seasonal_decompose(s, model=tipo, period=periodo)


def modelos(train, h, orden=(1, 1, 1), orden_estacional=(1, 1, 1), periodo=12):
    """Ajusta los 3 métodos (mismos parámetros por defecto que en clase) y pronostica h meses."""
    res = {}
    hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=periodo).fit()
    res["Holt-Winters"] = (hw.forecast(h), hw.resid)
    ar = ARIMA(train, order=orden).fit()
    res["ARIMA"] = (ar.forecast(h), ar.resid[orden[1]:])           # sin el primer residuo (diferenciación)
    sa = SARIMAX(train, order=orden, seasonal_order=(*orden_estacional, periodo)).fit(disp=False)
    res["SARIMA"] = (sa.forecast(h), sa.resid[periodo + 1:])
    return res


def metricas(real, pron):
    real, pron = np.asarray(real), np.asarray(pron)
    return {"MAE": mean_absolute_error(real, pron), "RMSE": np.sqrt(mean_squared_error(real, pron)),
            "MAPE_%": np.mean(np.abs((real - pron) / real)) * 100}


def evaluar(s, meses_prueba=12, **param):
    """Entrena con todo menos los últimos meses y compara contra lo real."""
    train, test = s.iloc[:-meses_prueba], s.iloc[-meses_prueba:]
    tabla, prueba = [], pd.DataFrame({"Real": test})
    for nombre, (f, resid) in modelos(train, meses_prueba, **param).items():
        prueba[nombre] = np.asarray(f)
        tabla.append({"Metodo": nombre, **metricas(test, f),
                      "LjungBox_p": acorr_ljungbox(resid, lags=[12])["lb_pvalue"].iloc[0]})
    return pd.DataFrame(tabla).sort_values("MAPE_%").reset_index(drop=True), prueba


def pronosticar(s, metodo, h=12, **param):
    f = pd.Series(np.asarray(modelos(s, h, **param)[metodo][0]),
                  index=pd.date_range(s.index[-1], periods=h + 1, freq="MS")[1:])
    nivel = np.select([f >= 94, f < 88], ["Alta", "Baja"], "Media")
    return pd.DataFrame({"Mes": f.index, "Pronostico": f.values, "Nivel_demanda": nivel})


if __name__ == "__main__":
    salidas = CARPETA / "salidas"
    salidas.mkdir(exist_ok=True)
    df = pd.read_csv(CARPETA / "data" / "reporte_cruceros_revenue_management.csv")
    s = serie_mensual(df)
    d = descomponer(s)
    adf = adfuller((s - d.seasonal).dropna())
    print(f"{len(s)} meses | fuerza estacional: el componente estacional va de {d.seasonal.min():.1f} a "
          f"{d.seasonal.max():.1f} puntos | ADF sin estacionalidad p = {adf[1]:.4f}")
    tabla, prueba = evaluar(s)
    print(tabla.round(3).to_string(index=False))
    mejor = tabla.loc[0, "Metodo"]
    pron = pronosticar(s, mejor)
    print(f"\nMejor método: {mejor}\n", pron.round(1).to_string(index=False))
    s.to_csv(salidas / "romero_serie.csv")
    pd.DataFrame({"Tendencia": d.trend, "Estacionalidad": d.seasonal, "Residuo": d.resid}).to_csv(salidas / "romero_descomposicion.csv")
    tabla.to_csv(salidas / "romero_metodos.csv", index=False)
    prueba.to_csv(salidas / "romero_prueba.csv")
    pron.to_csv(salidas / "romero_pronostico.csv", index=False)
