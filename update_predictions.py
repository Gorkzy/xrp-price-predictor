import numpy as np
import joblib
from tensorflow.keras.models import load_model
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Funkce pro získání aktuální ceny XRP
def get_current_xrp_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ripple&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        data = response.json()
        if "ripple" in data and "usd" in data["ripple"]:
            return float(data["ripple"]["usd"])
        else:
            return None
    except Exception as e:
        logging.error(f"Error fetching XRP price: {e}")
        return None

def update_prediction(model, scaler_X, scaler_y, window_size, model_type):
    current_price = get_current_xrp_price()
    if current_price is None:
        logging.error("Failed to fetch XRP price")
        return None
    current_price = round(current_price, 5)
    X_input = np.array([[current_price] * window_size])
    X_scaled = scaler_X.transform(X_input)
    prediction = model.predict(X_scaled)
    predicted_price_arr = scaler_y.inverse_transform(prediction)
    predicted_price = round(float(predicted_price_arr[0][0]), 5)
    logging.info(f"{datetime.now()}: {model_type} prediction: {predicted_price}")
    return {"current_price": current_price, "predicted_price": predicted_price, "model_type": model_type}

# Načti model a scalery pro krátkodobý model
model_short = load_model("xrp_model_short.h5", compile=False)
scaler_X_short = joblib.load("scaler_X_short.pkl")
scaler_y_short = joblib.load("scaler_y_short.pkl")

# Načti model a scalery pro dlouhodobý model
model_long = load_model("xrp_model_long.h5", compile=False)
scaler_X_long = joblib.load("scaler_X_long.pkl")
scaler_y_long = joblib.load("scaler_y_long.pkl")

# Aktualizuj predikci pro oba modely
short_prediction = update_prediction(model_short, scaler_X_short, scaler_y_short, window_size=5, model_type="short")
long_prediction = update_prediction(model_long, scaler_X_long, scaler_y_long, window_size=10, model_type="long")

# Ulož predikce do souboru (můžeš je také uložit do databáze)
import json
with open("latest_predictions.json", "w") as f:
    json.dump({"short": short_prediction, "long": long_prediction}, f)