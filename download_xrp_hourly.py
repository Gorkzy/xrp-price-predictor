import requests
import csv
from datetime import datetime

# URL pro stažení historických dat o hodinovém vývoji ceny XRP
url = "https://api.binance.com/api/v3/klines?symbol=XRPUSDT&interval=1h&limit=1000"
response = requests.get(url)

# Pokud API odpoví správně, získáme data jako JSON (každý záznam je seznam)
klines = response.json()

# Otevřeme CSV soubor pro zápis (přepíšeme jej, pokud již existuje)
with open("xrp_hourly.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Zapíšeme hlavičku CSV souboru
    writer.writerow(["datetime", "close_price"])
    
    # Pro každý záznam získáme čas a cenu zavření
    for kline in klines:
        open_time = int(kline[0])  # čas v milisekundách
        close_price = float(kline[4])
        # Převod času na čitelný formát (například "2025-02-10 12:00:00")
        dt = datetime.fromtimestamp(open_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
        # Zapíšeme řádek do CSV
        writer.writerow([dt, close_price])

print("Krátkodobá data uložena do xrp_hourly.csv")