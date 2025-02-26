import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

# Načtení dat
df = pd.read_csv("xrp_hourly.csv")

# Ujistěte se, že sloupec s cenou má správný název (např. "close_price")
# Pokud máte v CSV sloupec "close", přejmenujte jej:
df.rename(columns={"close": "close_price"}, inplace=True)

# Výpočet klouzavého průměru (SMA) s oknem 10
sma_indicator = SMAIndicator(close=df["close_price"], window=10)
df["SMA_10"] = sma_indicator.sma_indicator()

# Výpočet RSI s oknem 14
rsi_indicator = RSIIndicator(close=df["close_price"], window=14)
df["RSI_14"] = rsi_indicator.rsi()

# Výpočet MACD
macd_indicator = MACD(close=df["close_price"])
df["MACD"] = macd_indicator.macd()
df["MACD_signal"] = macd_indicator.macd_signal()

# Uložení dat s vypočtenými indikátory do nového CSV souboru
df.to_csv("xrp_hourly_with_indicators.csv", index=False)
print("Indikátory byly vypočteny a data uložena.")