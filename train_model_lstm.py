import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Načti data s indikátory (můžeš přizpůsobit, např. použij sloupec close_price a další indikátory)
df = pd.read_csv("xrp_hourly_with_indicators.csv")
prices = df["close_price"].values

# Definuj velikost okna (např. 10 hodin)
window_size = 10

X = []
y = []
for i in range(len(prices) - window_size):
    X.append(prices[i:i+window_size])
    y.append(prices[i+window_size])
X = np.array(X)
y = np.array(y)

# Normalizace
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Uprav tvar pro LSTM: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Vytvoření modelu
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Ulož model a scalery
model.save("xrp_model_lstm.h5")
joblib.dump(scaler_X, "scaler_X_lstm.pkl")
joblib.dump(scaler_y, "scaler_y_lstm.pkl")
print("LSTM model a scalery byly uloženy.")