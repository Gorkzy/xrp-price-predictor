import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib

# Načti data s indikátory (např. jen cenu)
df = pd.read_csv("xrp_hourly_with_indicators.csv")
# Pro ARIMA obvykle stačí časová řada – zde použijeme například sloupec close_price
price_series = df["close_price"]

# Vyber parametry ARIMA (p, d, q) – tyto hodnoty můžeš ladit
p, d, q = 1, 1, 1

model_arima = sm.tsa.ARIMA(price_series, order=(p, d, q))
model_arima_fit = model_arima.fit()

# Ulož ARIMA model
joblib.dump(model_arima_fit, "xrp_model_arima.pkl")
print("ARIMA model uložen.")