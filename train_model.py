import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Načtení dat z NumPy souborů
X = np.load("X.npy")
y = np.load("y.npy")

# Reshape X pro LSTM vrstvy (důležité!)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Rozdělení dat na trénovací a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vytvoření modelu s LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),  # input_shape upraveno
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),  # Další skrytá vrstva
    Dense(1)  # Výstupní vrstva pro predikci ceny
])

# Kompilace modelu
model.compile(optimizer='adam', loss='mean_squared_error')

# Trénink modelu
print("Začíná trénink modelu...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)

# Uložení modelu
model.save("xrp_model.h5")
print("Model byl úspěšně natrénován a uložen jako xrp_model.h5")
