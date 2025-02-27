import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Načti data s indikátory
df = pd.read_csv("xrp_hourly_with_indicators.csv")

# Vyber vstupní feature – například: close_price, SMA_10, RSI_14, MACD, MACD_signal
features = ["close_price", "SMA_10", "RSI_14", "MACD", "MACD_signal"]
X = df[features].values

# Předpokládej, že cílem je predikce budoucí uzavírací ceny
# Můžeme posunout data o 1 (předpovědět cenu následující hodiny)
y = np.roll(df["close_price"].to_numpy(), -1)[:-1]
X = X[:-1]  # odřízneme poslední řádek, protože y je posunuté

print("Tvar X:", X.shape)
print("Tvar y:", y.shape)

# Normalizace
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Vytvoř MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Trénuji MLP model založený na indikátorech...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# Ulož model a scalery
model.save("xrp_model_mlp.h5")
joblib.dump(scaler_X, "scaler_X_mlp.pkl")
joblib.dump(scaler_y, "scaler_y_mlp.pkl")
print("MLP model a scalery byly uloženy.")
