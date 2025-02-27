import pandas as pd
import ta

# Načti data – upravte cestu, pokud je potřeba
df = pd.read_csv("xrp_hourly.csv")

# Ujisti se, že sloupec s cenou má správný název; pokud máte sloupec "close", přejmenujte ho:
if "close" in df.columns:
    df.rename(columns={"close": "close_price"}, inplace=True)

# Vypočítej SMA (Simple Moving Average) pomocí třídy SMAIndicator
sma_indicator = ta.trend.SMAIndicator(close=df["close_price"], window=10)
df["SMA_10"] = sma_indicator.sma_indicator()

# Vypočítej RSI (Relative Strength Index) pomocí RSIIndicator
rsi_indicator = ta.momentum.RSIIndicator(close=df["close_price"], window=14)
df["RSI_14"] = rsi_indicator.rsi()

# Vypočítej MACD pomocí MACD třídy
macd_indicator = ta.trend.MACD(close=df["close_price"])
df["MACD"] = macd_indicator.macd()
df["MACD_signal"] = macd_indicator.macd_signal()

# Ulož data s indikátory
df.to_csv("xrp_hourly_with_indicators.csv", index=False)
print("Indikátory byly vypočteny a data uložena.")
