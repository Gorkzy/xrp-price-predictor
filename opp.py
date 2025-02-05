from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model  # type: ignore
import logging
import requests  # Import knihovny pro HTTP požadavky
from dotenv import load_dotenv

# Načti proměnné z .env souboru (pro lokální testování)
load_dotenv()

# Nastavení logování
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# Načtení API klíčů z environment variables
VALID_API_KEYS = set(os.getenv('VALID_API_KEYS', '').split(','))

# Debugging: Log počet načtených API klíčů (bez jejich obsahu)
logging.info(f"Number of API keys loaded: {len(VALID_API_KEYS)}")

# Funkce pro získání aktuální ceny XRP z Binance API
def get_current_xrp_price():
    """Získá aktuální cenu XRP z Binance API nebo alternativního API."""
    binance_url = "https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT"

    try:
        response = requests.get(binance_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        current_price = float(data['price'])
        logging.info(f"✅ XRP cena z Binance: {current_price}")
        return current_price
    except requests.exceptions.HTTPError as e:
        logging.error(f"❌ HTTP chyba Binance API: {e.response.status_code} {e.response.reason}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Jiná chyba při přístupu k Binance API: {e}")
    
    return None  # Pokud selže Binance API

# Načtení modelu
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "xrp_model.h5")

if not os.path.exists(model_path):
    logging.error(f"❌ Model file not found at {model_path}")
    raise FileNotFoundError("Model file not found")

try:
    model = load_model(model_path, compile=False)
    logging.info(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    raise

# Inicializace aplikace Flask
app = Flask(__name__)

# Endpoint pro automatickou predikci
@app.route("/", methods=["GET"])
def home():
    logger.debug("Root endpoint accessed")
    return "Flask server is running!", 200 

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pro zpracování požadavku na predikci."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON in request'}), 400

    api_key = data.get('api_key')

    if api_key not in VALID_API_KEYS:
        logging.warning(f"⚠ Neplatný API klíč: {api_key}")
        return jsonify({'error': 'Invalid API key'}), 401

    current_price = get_current_xrp_price()

    if current_price is None:
        return jsonify({'error': 'Could not fetch current XRP price'}), 500

    X = np.array([[current_price, current_price]])

    try:
        prediction = model.predict(X)
        predicted_price = float(prediction[0][0])
        logging.info(f"✅ Prediction successful: current_price={current_price}, predicted_price={predicted_price}")
        return jsonify({'current_price': current_price, 'predicted_price': predicted_price})

    except Exception as e:
        logging.error(f"❌ Chyba při predikci: {e}")
        return jsonify({'error': 'Prediction error'}), 500

# Endpoint pro hlavní stránku
@app.route('/')
def index():
    return render_template('index.html')

# Spuštění aplikace
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Google Cloud Run používá proměnnou PORT
    logging.info(f"🚀 Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=8080, debug=True)