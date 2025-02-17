import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib


# Načtení dat
X = np.load("X.npy")
y = np.load("y.npy")


df = pd.read_csv("xrp_daily.csv")
prices = df["price"].values  # předpokládej, že sloupec se jmenuje "price"
window_size = 10  # například 10 dní
X = []
y = []
for i in range(len(prices) - window_size):
    X.append(prices[i:i+window_size])
    y.append(prices[i+window_size])
X = np.array(X)
y = np.array(y)
# Kontrola tvaru X
print("Tvar X:", X.shape)

# Normalizace dat
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)) # y musí být matice pro scaler

# Rozdělení dat na trénovací a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Vytvoření modelu - UPRAVENO PRO 2D DATA
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Input shape se automaticky přizpůsobí
    Dense(64, activation='relu'),
    Dense(1)
])

# Kompilace modelu
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Trénink modelu
print("Začíná trénink modelu...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=8)

# Uložení modelu
model.save("xrp_model_long.h5")
print("Model uložen.")

# Uložení scalerů
joblib.dump(scaler_X, "scaler_X_long.pkl")
joblib.dump(scaler_y, "scaler_y_long.pkl")
print("Dlouhodobý model a scalery byly uloženy.")