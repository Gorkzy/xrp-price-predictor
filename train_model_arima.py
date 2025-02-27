import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib
from sklearn.preprocessing import MinMaxScaler

# Načti data s indikátory (např. jen cenu)
df = pd.read_csv("xrp_hourly_with_indicators.csv")

# Pro ARIMA obvykle stačí časová řada – zde použijeme sloupec close_price
price_series = df["close_price"].to_numpy().reshape(-1, 1)

# Vytvoř scaler a transformuj data
scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(price_series)

# ARIMA model od statsmodels pracuje s 1D polem, proto data zploštíme
price_scaled_series = price_scaled.flatten()

# Vyber parametry ARIMA (p, d, q) – tyto hodnoty můžeš ladit podle potřeby
p, d, q = 1, 1, 1

# Vytvoř a natrénuj ARIMA model na skalovaných datech
model_arima = sm.tsa.ARIMA(price_scaled_series, order=(p, d, q))
model_arima_fit = model_arima.fit()

# Ulož ARIMA model a scaler(y) – pro jednoduchost použijeme stejný scaler pro vstup i výstup
joblib.dump(model_arima_fit, "xrp_model_arima.pkl")
joblib.dump(scaler, "scaler_X_arima.pkl")
joblib.dump(scaler, "scaler_y_arima.pkl")

print("ARIMA model a scalery byly uloženy.")