import numpy as np
from tensorflow.keras.models import load
from sklearn.metrics import mean_absolute_error

# Načtení zpracovaných dat
X = np.load("X.npy")
y = np.load("y.npy")

# Rozdělení na trénovací a testovací data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Načtení natrénovaného modelu
model = load_model("xrp_model.h5")

# Predikce na testovacích datech
y_pred = model.predict(X_test)

# Výpočet chyby (MAE - Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
print(f"Průměrná absolutní chyba predikce: {mae:.4f}")

# Ukázka skutečných a predikovaných hodnot
print("\nUkázka predikcí:")
for actual, predicted in zip(y_test[:5], y_pred[:5]):
    print(f"Skutečná hodnota: {actual:.4f}, Predikovaná hodnota: {predicted[0]:.4f}")