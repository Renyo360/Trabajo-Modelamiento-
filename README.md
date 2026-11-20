# Net Yield de una flota de cruceros – Modelamiento Predictivo de Datos

KPI central: **Net Yield** = ingreso neto ÷ cabina-noche disponible. Hoy US$ 305.0 → meta US$ 320.2 (+5 %).

| Archivo | Integrante | Qué hace |
|---|---|---|
| `romero_series_tiempo.py` | Romero | Descomposición + Holt-Winters, ARIMA(1,1,1) y SARIMA(1,1,1)(1,1,1)12; prueba de 12 meses (MAE, RMSE, MAPE, Ljung-Box) y pronóstico de 12 meses |
| `rubina_revenue_rf.py` | Rubina | Random Forest del gasto a bordo (R², RMSE, MAE, OOB + vista de clasificación), tarifa óptima por temporada y sensibilidad a la elasticidad |
| `bustamante_simulacion.py` | Bustamante | Simulación de reservas (SimPy), escenarios A/B/C con 30 réplicas y tabla de parámetros para Arena |
| `diaz_streamlit.py` | Díaz | Simulador web: sube el CSV y mueve los parámetros de los tres modelos |

## Cómo correrlo

```bash
pip install -r requirements.txt
python romero_series_tiempo.py      # ~5 s
python rubina_revenue_rf.py         # ~25 s
python bustamante_simulacion.py     # ~10 s (usa la tarifa de Rubina)
streamlit run diaz_streamlit.py     # abre el simulador en el navegador
```

Los tres scripts guardan sus resultados en `salidas/`. El simulador no necesita esos archivos: llama a las mismas funciones y recalcula todo con los parámetros que elijas.

## Publicar el simulador (para el QR del póster)

1. Sube esta carpeta a un repositorio público de GitHub (con `data/` incluida).
2. Entra a share.streamlit.io → *Create app* → elige el repositorio y el archivo `diaz_streamlit.py`.
3. Copia la URL que te da, pégala en el recuadro del informe y genera el QR para el póster.
