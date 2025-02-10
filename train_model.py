import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model  # Pro načtení modelu
from sklearn.preprocessing import MinMaxScaler  # Pro normalizaci dat

# Načtení dat z NumPy souborů
try:
    X_original = np.load("X.npy")
    y_original = np.load("y.npy")
    print("Data X.npy a y.npy načtena.")
except FileNotFoundError:
    print("Chyba: Soubory X.npy nebo y.npy nebyly nalezeny. Ujistěte se, že jsou ve stejném adresáři jako skript.")
    exit()
except Exception as e:
    print(f"Chyba při načítání dat: {e}")
    exit()

# Upravená funkce pro vytvoření sekvencí pro hodinovou predikci
def create_sequences(data, seq_length, prediction_horizon):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - prediction_horizon + 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length + prediction_horizon - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# **ZDE JSOU KLÍČOVÉ ZMĚNY:**
sequence_length = 30  # Zmenšeno na 30
prediction_horizon = 1 # Zmenšeno na 1 (predikce o jeden krok napřed)

# Vytvoření sekvencí pomocí upravené funkce
X, y = create_sequences(X_original, sequence_length, prediction_horizon)

# Výběr první feature z X (pokud máš více než jednu feature)
if X.ndim == 3:  # Kontrola, zda má X 3 rozměry
    X = X[:, :, 0]  # Vezmeme všechny sekvence, všechny časové kroky, ale jen první feature

# Kontrola tvaru X a y a jejich obsahu
print("Tvar X před reshape:", X.shape)
print("Tvar y:", y.shape)

if X.size == 0 or y.size == 0:
    print("Chyba: Pole X nebo y je prázdné. Zkontrolujte prosím data a parametry sekvencí.")
    print(f"Délka X_original: {len(X_original)}, seq_length: {sequence_length}, prediction_horizon: {prediction_horizon}")
    print(f"Délka y_original: {len(y_original)}, seq_length: {sequence_length}, prediction_horizon: {prediction_horizon}")
    exit()

# Reshape X pro LSTM vrstvy (důležité!)
X = X.reshape(X.shape[0], X.shape[1], 1)
print("Tvar X po reshape:", X.shape)

# Rozdělení dat na trénovací a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Vytvoření modelu s LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Kompilace modelu
model.compile(optimizer='adam', loss='mean_squared_error')

# Trénink modelu
print("Začíná trénink modelu...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)

# Uložení modelu
model.save("xrp_model.h5")
print("Model byl úspěšně natrénován a uložen jako xrp_model.h5")