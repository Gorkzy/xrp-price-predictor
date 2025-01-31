import requests
from datetime import datetime
import csv

# Binance API endpoint
url = "https://api.binance.com/api/v3/klines"

# Parametry pro získání hodinových cen XRP/USDT
params = {
    "symbol": "XRPUSDT",
    "interval": "1h",
    "limit": 100  # Počet posledních záznamů
}

try:
    # Pošleme požadavek na Binance API
    response = requests.get(url, params=params)
    response.raise_for_status()  # Zkontroluje chyby

    # Získáme data ve formátu JSON
    data = response.json()

    # Vybereme jen relevantní data (čas, otevírací cena, zavírací cena)
    prices = [
        {
            "time": datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            "open": float(item[1]),
            "close": float(item[4])
        }
        for item in data
    ]

    # Uložíme data do CSV souboru
    with open("xrp_prices.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["time", "open", "close"])
        writer.writeheader()  # Zapisujeme hlavičku
        writer.writerows(prices)  # Zapisujeme data

    print("Data byla úspěšně uložena do souboru xrp_prices.csv")

except requests.exceptions.RequestException as e:
    print(f"Chyba při získávání dat: {e}")