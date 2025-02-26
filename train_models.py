import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

### TRÉNINK KRÁTKODOBÉHO MODELU (HODINOVÁ DATA)
print("=== Trénink krátkodobého modelu ===")
# Předpokládáme, že soubor xrp_hourly.csv obsahuje sloupec "close_price"
df_hourly = pd.read_csv("xrp_hourly.csv")
prices_hourly = df_hourly["close"].values

window_size_short = 5  # např. 5 hodin
X_short = []
y_short = []
for i in range(len(prices_hourly) - window_size_short):
    X_short.append(prices_hourly[i:i + window_size_short])
    y_short.append(prices_hourly[i + window_size_short])
X_short = np.array(X_short)
y_short = np.array(y_short)

print("Tvar X (krátkodobý):", X_short.shape)

# Normalizace
scaler_X_short = MinMaxScaler()
X_short_scaled = scaler_X_short.fit_transform(X_short)
scaler_y_short = MinMaxScaler()
y_short_scaled = scaler_y_short.fit_transform(y_short.reshape(-1, 1))

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X_short_scaled, y_short_scaled, test_size=0.2, random_state=42)

# Vytvoření modelu
model_short = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_short.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Trénuji krátkodobý model...")
history_short = model_short.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)

# Uložení modelu a scalerů
model_short.save("xrp_model_short.h5")
joblib.dump(scaler_X_short, "scaler_X_short.pkl")
joblib.dump(scaler_y_short, "scaler_y_short.pkl")
print("Krátkodobý model a scalery byly uloženy.\n")


### TRÉNINK DLOUHODOBÉHO MODELU (DENNÍ DATA)
print("=== Trénink dlouhodobého modelu ===")
# Předpokládáme, že soubor xrp_daily.csv obsahuje sloupec "price"
df_daily = pd.read_csv("xrp_daily.csv")
prices_daily = df_daily["price"].values

window_size_long = 10  # např. 10 dní
X_long = []
y_long = []
for i in range(len(prices_daily) - window_size_long):
    X_long.append(prices_daily[i:i + window_size_long])
    y_long.append(prices_daily[i + window_size_long])
X_long = np.array(X_long)
y_long = np.array(y_long)

print("Tvar X (dlouhodobý):", X_long.shape)

# Normalizace
scaler_X_long = MinMaxScaler()
X_long_scaled = scaler_X_long.fit_transform(X_long)
scaler_y_long = MinMaxScaler()
y_long_scaled = scaler_y_long.fit_transform(y_long.reshape(-1, 1))

# Rozdělení dat
X_train_long, X_test_long, y_train_long, y_test_long = train_test_split(X_long_scaled, y_long_scaled, test_size=0.2, random_state=42)

# Vytvoření modelu – lze vyzkoušet mírně složitější architekturu
model_long = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_long.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model_long.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Trénuji dlouhodobý model...")
history_long = model_long.fit(X_train_long, y_train_long, validation_data=(X_test_long, y_test_long), epochs=100, batch_size=8)

# Uložení modelu a scalerů
model_long.save("xrp_model_long.h5")
joblib.dump(scaler_X_long, "scaler_X_long.pkl")
joblib.dump(scaler_y_long, "scaler_y_long.pkl")
print("Dlouhodobý model a scalery byly uloženy.")