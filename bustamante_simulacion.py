"""
BUSTAMANTE - Simulación del proceso de reservas (Python; el mismo flujo se arma en Arena)
========================================================================================
Simula la venta de un zarpe de 7 noches durante 180 días (por defecto el Icon of the Seas, 2,805 cabinas).
Cada solicitud de reserva:  llega  ->  ¿hay cabinas?  ->  ¿acepta la tarifa?  ->  venta (suma ingreso neto).
Escenarios:
  A  Actual        tarifa fija por temporada
  B  RM            tarifa óptima de Rubina
  C  RM + directo  tarifa de Rubina y campaña de venta directa (reservas por agencia B2B: 44 % -> 30 %)
30 réplicas por escenario y temporada, con la misma semilla en A, B y C (comparación justa).
Ejecutar (después de Rubina):  python bustamante_simulacion.py
"""
import numpy as np
import pandas as pd
import simpy
from scipy import stats

from rubina_revenue_rf import COMISION_B2B, ELASTICIDAD, MARGEN_BORDO, SALIDAS, TARIFA_ACTUAL, cargar_datos, tabla_zarpes

NOCHES, DIAS_VENTA, CONVERSION, REPLICAS = 7, 180, 0.60, 30
P_2_CABINAS = 0.151                       # 15.1 % de reservas toma 2 cabinas (dataset)
P_HUESPEDES = [0.500, 0.479, 0.021]       # 1, 2 o 3 huéspedes por cabina (dataset)


def parametros(df, tarifas_rm, barco="Icon of the Seas", conversion=CONVERSION, b2b_c=0.30, dias=DIAS_VENTA):
    """Análisis de entrada: ocupación, gasto a bordo y % B2B del barco elegido, por temporada."""
    z = tabla_zarpes(df)
    zb, db = z[z["Barco"] == barco], df[df["Barco"] == barco]
    capacidad = int(zb["Capacidad"].iloc[0])
    b2b = float((db["Booking_Source"] == "B2B").mean())
    par = {"barco": barco, "capacidad": capacidad, "conversion": conversion, "dias": dias,
           "b2b": {"A": b2b, "B": b2b, "C": b2b_c}}
    for t in ["Alta", "Media", "Baja"]:
        gasto = db.loc[db["Temporada"] == t, "Gasto_Promedio_Diario_Huesped_USD"]
        ocup = zb.loc[zb["Temporada"] == t, "Ocupacion"].mean()
        solicitudes = ocup * capacidad / (1 + P_2_CABINAS) / conversion       # solicitudes que llegan por zarpe
        par[t] = {"ocupacion_hist": ocup, "ny_hist": zb.loc[zb["Temporada"] == t, "Net_Yield"].mean(),
                  "entre_llegadas": dias / solicitudes, "gasto": (gasto.mean(), gasto.std()),
                  "ks_p": stats.kstest(gasto.to_numpy(), stats.norm(gasto.mean(), gasto.std()).cdf).pvalue,
                  "tarifa": {"A": TARIFA_ACTUAL[t], "B": tarifas_rm[t], "C": tarifas_rm[t]}}
    pesos = z.groupby("Temporada")["Cab_Noches_Disp"].sum()
    return par, (pesos / pesos.sum()).to_dict()


def prob_compra(par, t, esc, elasticidad=ELASTICIDAD):
    """Probabilidad de aceptar la tarifa: 60 % a la tarifa actual, sube si la tarifa baja."""
    return min(1.0, par["conversion"] * (par[t]["tarifa"][esc] / TARIFA_ACTUAL[t]) ** (-elasticidad))


def simular_zarpe(par, t, esc, semilla, elasticidad=ELASTICIDAD):
    rng = np.random.default_rng(semilla)
    env = simpy.Environment()
    cap, tarifa, p = par["capacidad"], par[t]["tarifa"][esc], prob_compra(par, t, esc, elasticidad)
    r = {"vendidas": 0, "ingreso": 0.0, "solicitudes": 0, "rechazadas": 0}

    def solicitudes():
        while True:
            yield env.timeout(rng.exponential(par[t]["entre_llegadas"]))       # Create
            r["solicitudes"] += 1
            cabinas = 2 if rng.random() < P_2_CABINAS else 1                   # Assign
            u = rng.random()
            huespedes = cabinas * (1 if u < P_HUESPEDES[0] else 2 if u < P_HUESPEDES[0] + P_HUESPEDES[1] else 3)
            gasto = rng.normal(*par[t]["gasto"])
            comision = COMISION_B2B if rng.random() < par["b2b"][esc] else 0
            if r["vendidas"] + cabinas > cap:                                   # Decide: ¿hay cabinas?
                r["rechazadas"] += 1
                continue
            if rng.random() > p:                                                # Decide: ¿acepta la tarifa?
                continue
            r["vendidas"] += cabinas                                            # Assign + Record
            r["ingreso"] += cabinas * NOCHES * tarifa * (1 - comision) + huespedes * NOCHES * gasto * MARGEN_BORDO

    env.process(solicitudes())
    env.run(until=par["dias"])
    return {"Temporada": t, "Escenario": esc, "Replica": semilla, "Ocupacion": r["vendidas"] / cap,
            "Net_Yield": r["ingreso"] / (cap * NOCHES), "Rechazadas_pct": r["rechazadas"] / r["solicitudes"]}


def correr(par, pesos, replicas=REPLICAS, elasticidad=ELASTICIDAD):
    sim = pd.DataFrame([simular_zarpe(par, t, e, s, elasticidad)
                        for t in ["Alta", "Media", "Baja"] for e in "ABC" for s in range(replicas)])
    sim["Peso"] = sim["Temporada"].map(pesos)
    anual = sim.groupby(["Escenario", "Replica"]).apply(
        lambda x: pd.Series({"Net_Yield": np.average(x["Net_Yield"], weights=x["Peso"]),
                             "Ocupacion": np.average(x["Ocupacion"], weights=x["Peso"])}), include_groups=False)
    res = anual.groupby("Escenario").agg(Net_Yield=("Net_Yield", "mean"), sd=("Net_Yield", "std"),
                                         Ocupacion=("Ocupacion", "mean"))
    res["IC95"] = stats.t.ppf(0.975, replicas - 1) * res.pop("sd") / np.sqrt(replicas)
    res["Mejora_vs_A"] = res["Net_Yield"] / res.loc["A", "Net_Yield"] - 1
    res["p_valor"] = [np.nan] + [stats.ttest_rel(anual.loc[e, "Net_Yield"], anual.loc["A", "Net_Yield"]).pvalue
                                 for e in "BC"]
    por_temp = sim.groupby(["Temporada", "Escenario"])[["Net_Yield", "Ocupacion", "Rechazadas_pct"]].mean().reset_index()
    val = por_temp[por_temp["Escenario"] == "A"].set_index("Temporada")[["Ocupacion", "Net_Yield"]]
    val.insert(0, "Ocupacion_real", [par[t]["ocupacion_hist"] for t in val.index])
    val.insert(2, "NY_real", [par[t]["ny_hist"] for t in val.index])
    val["Error_NY_%"] = (val["Net_Yield"] / val["NY_real"] - 1) * 100
    return res.reset_index(), por_temp, val.reset_index()


def tabla_arena(par, elasticidad=ELASTICIDAD):
    """Valores para cargar en los módulos de Arena (unidad de tiempo: horas)."""
    filas = []
    for t in ["Alta", "Media", "Baja"]:
        v = par[t]
        for e in "ABC":
            filas.append({"Temporada": t, "Escenario": e,
                          "Create: entre llegadas (h)": f"EXPO({v['entre_llegadas'] * 24:.2f})",
                          "Assign: Tarifa": round(v["tarifa"][e], 1),
                          "Assign: Gasto diario": f"NORM({v['gasto'][0]:.1f}, {v['gasto'][1]:.1f})",
                          "Assign: Comision": f"DISC({par['b2b'][e]:.3f}, 0.15, 1.0, 0)",
                          "Decide: % acepta tarifa": round(prob_compra(par, t, e, elasticidad) * 100, 1)})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = cargar_datos()
    tarifas_rm = pd.read_csv(SALIDAS / "rubina_tarifas.csv").set_index("Temporada")["Tarifa_optima"].to_dict()
    par, pesos = parametros(df, tarifas_rm)
    for t in ["Alta", "Media", "Baja"]:
        v = par[t]
        print(f"{t}: ocupación {v['ocupacion_hist']:.1%} | gasto NORM({v['gasto'][0]:.1f}, {v['gasto'][1]:.1f}) "
              f"KS p = {v['ks_p']:.2f} | entre llegadas {v['entre_llegadas'] * 24:.2f} h")
    res, por_temp, val = correr(par, pesos)
    print("\nValidación (A vs histórico):\n", val.round(3).to_string(index=False))
    print("\nPor temporada:\n", por_temp.round(4).to_string(index=False))
    print("\nAnual:\n", res.round(4).to_string(index=False))
    res.to_csv(SALIDAS / "bustamante_escenarios.csv", index=False)
    por_temp.to_csv(SALIDAS / "bustamante_por_temporada.csv", index=False)
    val.to_csv(SALIDAS / "bustamante_validacion.csv", index=False)
    tabla_arena(par).to_csv(SALIDAS / "bustamante_parametros_arena.csv", index=False, encoding="utf-8-sig")
