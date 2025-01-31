import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Načtení dat
X = np.load("X.npy")
y = np.load("y.npy")

# Rozdělení dat na trénovací a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vytvoření modelu
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),  # Skrytá vrstva
    Dense(64, activation='relu'),                       # Další skrytá vrstva
    Dense(1)                                            # Výstupní vrstva
])

# Kompilace modelu
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Trénink modelu
print("Začíná trénink modelu...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)

# Uložení modelu
model.save("xrp_model.h5")
print("Model byl úspěšně natrénován a uložen jako xrp_model.h5")