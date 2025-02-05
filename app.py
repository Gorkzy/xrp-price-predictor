from flask import Flask, render_template, request, jsonify
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU (optional)
from tensorflow.keras.models import load_model  # type: ignore
import logging
import requests  # Import knihovny pro HTTP požadavky
from dotenv import load_dotenv

# Načti proměnné z .env souboru (pro lokální testování)
load_dotenv()

# Nastavení logování
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Načtení API klíčů z environment variables
VALID_API_KEYS = set(os.getenv('VALID_API_KEYS', '').split(','))

# Debugging: Log počet načtených API klíčů (bez jejich obsahu)
logging.info(f"Number of API keys loaded: {len(VALID_API_KEYS)}")

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

        logging.debug(f"Binance API response: {data}")  # Debug log

        if "price" in data:
            current_price = float(data["price"])
            logging.info(f"Current XRP price fetched: {current_price}")
            return current_price
        else:
            logging.error(f"API response does not contain 'price' key! Response: {data}")
            return None

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

# Endpoint pro automatickou predikci
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pro zpracování požadavku na predikci.

    Očekává POST požadavek s API klíčem v těle požadavku ve formátu JSON.
    """
    # Získání API klíče z požadavku
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON in request'}), 400

    api_key = data.get('api_key')

    # Ověření platnosti API klíče
    if api_key not in VALID_API_KEYS:
        logging.warning(f"Invalid API key: {api_key}")
        return jsonify({'error': 'Invalid API key'}), 401

    # Získání aktuální ceny XRP
    current_price = get_current_xrp_price()

    if current_price is None:
        return jsonify({'error': 'Could not fetch current XRP price'}), 500

    try:
        # Vytvoření vstupu pro model (použijeme stejnou hodnotu pro 'open' a 'close')
        X = np.array([[current_price, current_price]])

        # Predikce
        prediction = model.predict(X)
        predicted_price = float(prediction[0][0])

        # Návrat výsledku jako JSON
        logging.info(f"Prediction successful: current_price={current_price}, predicted_price={predicted_price}")
        return jsonify({'current_price': current_price, 'predicted_price': predicted_price})
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

# Definování cesty k šabloně
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)