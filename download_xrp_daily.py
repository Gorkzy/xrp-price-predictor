import requests
import csv
from datetime import datetime

url = "https://api.coingecko.com/api/v3/coins/ripple/market_chart?vs_currency=usd&days=365"
response = requests.get(url)
data = response.json()

# Data obsahují položku "prices" – jedná se o seznam dvojic [timestamp, cena]
prices = data.get("prices", [])

# Uložení do CSV souboru
with open("xrp_daily.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["date", "price"])
    
    for timestamp, price in prices:
        # Převeď timestamp (v milisekundách) na čitelný datum
        dt = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
        writer.writerow([dt, price])

print("Dlouhodobá data uložena do xrp_daily.csv")