import pandas as pd
import numpy as np

# Načtení dat z CSV
data = pd.read_csv("xrp_prices.csv")

# Zobrazení prvních pár řádků pro kontrolu
print("Ukázka dat:")
print(data.head())

# Vytvoření vstupů (features) a výstupů (labels)
# Použijeme 'close' jako hodnotu, kterou predikujeme
data["next_close"] = data["close"].shift(-1)  # Posuneme 'close' o jeden řádek nahoru
data = data[:-1]  # Poslední řádek odstraníme, protože nemá 'next_close'

X = data[["open", "close"]].values  # Vstupy: otevření a uzavření
y = data["next_close"].values       # Výstupy: další uzavírací cena

print("\nPřipravená data:")
print(f"Vstupy (X): {X[:5]}")
print(f"Výstupy (y): {y[:5]}")

# Uložení zpracovaných dat (volitelné)
np.save("X.npy", X)
np.save("y.npy", y)
print("\nData byla úspěšně zpracována a uložena.")