import requests
import csv
from datetime import datetime

# URL pro získání historických dat z Binance API (XRPUSDT, interval 1h, limit 1000 záznamů)
url = "https://api.binance.com/api/v3/klines?symbol=XRPUSDT&interval=1h&limit=1000"
response = requests.get(url)
klines = response.json()

# Otevřeme soubor xrp_hourly.csv pro zápis dat
with open("xrp_hourly.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Zapíšeme hlavičku CSV souboru
    writer.writerow(["datetime", "open", "high", "low", "close", "volume"])
    # Pro každý záznam v datech z API získáme potřebné hodnoty
    for kline in klines:
        open_time = int(kline[0])
        open_price = float(kline[1])
        high = float(kline[2])
        low = float(kline[3])
        close = float(kline[4])
        volume = float(kline[5])
        # Převedeme čas z milisekund na čitelný formát
        dt = datetime.fromtimestamp(open_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([dt, open_price, high, low, close, volume])

print("Data byla uložena do xrp_hourly.csv")