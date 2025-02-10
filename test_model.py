import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_absolute_error
import full_app
from sklearn.model_selection import train_test_split # Doplněný import

# Načtení dat
X = np.load("X.npy")
y = np.load("y.npy")

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Načtení modelu a scalerů
model = load_model("xrp_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Zkontrolujeme tvar X
print("Tvar X:", X.shape)

# Predikce na testovacích datech
X_test_scaled = scaler_X.transform(X_test)
y_pred_scaled = model.predict(X_test_scaled)

y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Výpočet MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Průměrná absolutní chyba predikce: {mae:.4f}")

# Ukázka predikcí
print("\nUkázka predikcí:")
for actual, predicted in zip(y_test[:5], y_pred[:5]):
    print(f"Skutečná hodnota: {actual:.4f}, Predikovaná hodnota: {predicted[0]:.4f}")

# Predikce aktuální ceny
aktualni_cena = full_app.get_current_xrp_price()

nova_data = X[-30:]
nova_data_scaled = scaler_X.transform(nova_data)

predikce_scaled = model.predict(nova_data_scaled)
predikce = scaler_y.inverse_transform(predikce_scaled)

print(f"\nAktuální cena XRP: {aktualni_cena}")
print(f"Predikce ceny XRP za hodinu: {predikce[0, 0]:.4f}")