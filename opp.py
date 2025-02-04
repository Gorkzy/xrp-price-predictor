from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model  # type: ignore
import logging
import requests
from dotenv import load_dotenv

# Načti proměnné z .env souboru (pro lokální testování)
load_dotenv()

# Nastavení logování
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Načtení API klíčů z environment variables
VALID_API_KEYS = set(os.getenv('VALID_API_KEYS', '').split(','))

# Debugging: Log počet načtených API klíčů
logging.info(f"🔑 Number of API keys loaded: {len(VALID_API_KEYS)}")

# Funkce pro získání aktuální ceny XRP z Binance API
def get_current_xrp_price():
    """Získá aktuální cenu XRP z Binance API."""
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
    
    return None  # Musíme vrátit None, pokud dojde k chybě

# Přidání testovacího endpointu pro Binance API
@app.route('/test_binance', methods=['GET'])
def test_binance():
    """Otestuje, zda Render.com může přistupovat k Binance API."""
    price = get_current_xrp_price()
    if price is None:
        return jsonify({"success": False, "error": "Binance API blocked or unreachable"}), 500
    return jsonify({"success": True, "price": price})

# Nastavení cesty a načtení modelu
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
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pro zpracování požadavku na predikci."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON in request'}), 400

    api_key = data.get('api_key')

    # Ověření platnosti API klíče
    if api_key not in VALID_API_KEYS:
        logging.warning(f"⚠ Neplatný API klíč: {api_key}")
        return jsonify({'error': 'Invalid API key'}), 401

    # Získání aktuální ceny XRP
    current_price = get_current_xrp_price()

    if current_price is None:
        return jsonify({'error': 'Could not fetch current XRP price'}), 500

    # Vytvoření vstupu pro model (použijeme stejnou hodnotu pro 'open' a 'close')
    X = np.array([[current_price, current_price]])

    # Predikce
    try:
        prediction = model.predict(X)
        predicted_price = float(prediction[0][0])

        # Návrat výsledku jako JSON
        logging.info(f"✅ Prediction successful: current_price={current_price}, predicted_price={predicted_price}")
        return jsonify({'current_price': current_price, 'predicted_price': predicted_price})

    except Exception as e:
        logging.error(f"❌ Chyba při predikci: {e}")
        return jsonify({'error': 'Prediction error'}), 500

# Definování cesty k šabloně
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    logging.info("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)