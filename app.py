from flask import Flask, jsonify, request
import numpy as np
import os
from tensorflow.keras.models import load_model  # type: ignore
import logging
import requests  # Import knihovny pro HTTP požadavky

# Nastavení logování
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def my_function():
    logging.info("Starting function...")
    # ... nějaký kód
    logging.info("Function finished.")

# Platné API klíče
VALID_API_KEYS = {
    "maxim-pidaras6944",
    "tom-mimon22",
    "premium1-e5d1b9a4f7c6e3f",
    "misa-auditt22",
    "user4-9c6a4f7e2d1b5e8c",
    "user1-8e3b5c6d9f1a4b7e",
    "user2-4f9e6a7d8c2b1e3f",
    "user3-5d8c1a4b7f9e2e6d",
    "admin-a1b2c3d4e5f6g7h8",
    "vip1-2e4d6f8a1b9c7e3f",
    "test1-3e7c9a2d4f1b5e8f",
    "guest1-6f1a9e3b7c2d5e4f",
    "demo1-8c4f7e9a1b5d2e6f"
}

# Funkce pro získání aktuální ceny XRP z Binance API
def get_current_xrp_price():
    """Fetches the current XRP price from the Binance API.

    Returns:
        float: The current XRP price, or None if an error occurs.
    """
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT"
        response = requests.get(url)
        data = response.json()
        current_price = float(data['price'])
        logging.info(f"Current XRP price fetched: {current_price}")
        return current_price
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching XRP price: {e}")
        return None
    except ValueError as e:
        logging.error(f"Invalid data format: {e}")
        return None
    except Exception as e:  # Zachytí všechny ostatní výjimky
        logging.error(f"Unexpected error: {e}")
        return None

# Nastavení cesty a načtení modelu
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "xrp_model.h5")

if not os.path.exists(model_path):
    logging.error(f"Model file not found at {model_path}")
    raise FileNotFoundError("Model file not found")

try:
    model = load_model(model_path, compile=False)
    logging.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Inicializace aplikace Flask
app = Flask(__name__)

# Middleware pro kontrolu API klíče
@app.before_request
def check_api_key():
    api_key = request.headers.get("x-api-key")  # Klíč se očekává v hlavičce požadavku
    logging.info(f"Received API key: {api_key}")
    if api_key not in VALID_API_KEYS:
        logging.warning(f"Unauthorized access attempt with API key: {api_key}")
        return jsonify({"error": "Invalid or missing API key"}), 403

# Endpoint pro automatickou predikci
@app.route('/predict', methods=['GET'])
def predict():
    try:
        logging.info("Received /predict request")
        
        # Získání aktuální ceny XRP
        current_price = get_current_xrp_price()

        if current_price is None:
            return jsonify({'error': 'Could not fetch current XRP price'}), 500

        # Vytvoření vstupu pro model (použijeme stejnou hodnotu pro 'open' a 'close')
        X = np.array([[current_price, current_price]])

        # Predikce
        prediction = model.predict(X)
        predicted_price = float(prediction[0][0])  # Převod na standardní float

        # Návrat výsledku jako JSON
        logging.info(f"Prediction successful: current_price={current_price}, predicted_price={predicted_price}")
        return jsonify({
            'current_price': current_price,
            'predicted_close': predicted_price
        })

    except Exception as e:
        logging.error(f"Error in /predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask server...")
    app.run(host="0.0.0.0", port=8080, debug=True)
