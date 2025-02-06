from flask import Flask, render_template, request, jsonify
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model  # type: ignore
import logging
import requests  # Import knihovny pro HTTP požadavky
from google.cloud import storage  # Google Cloud Storage
from dotenv import load_dotenv
import traceback
from google.cloud import error_reporting

# Načti proměnné z .env souboru (pro lokální testování)
load_dotenv()

# Nastavení logování
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Načtení API klíčů z environment variables
VALID_API_KEYS = os.environ.get('VALID_API_KEYS')
if VALID_API_KEYS is None:
    logging.error("VALID_API_KEYS is not set in environment variables!")
else:
    VALID_API_KEYS = set(filter(None, VALID_API_KEYS.split(',')))
    logging.info(f"✅ Načtené API klíče: {VALID_API_KEYS}")

# Google Cloud Storage - Bucket a soubor modelu
BUCKET_NAME = "xrp-model-storage"  # Změň na skutečný název bucketu
MODEL_FILE = "xrp_model.h5"
LOCAL_MODEL_PATH = "/tmp/xrp_model.h5"  # App Engine umožňuje zápis pouze do `/tmp`

def download_model():
    """Stáhne model z Google Cloud Storage do dočasného souboru."""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        logging.info(f"✅ Model stažen do {LOCAL_MODEL_PATH}")
    except Exception as e:
        logging.error(f"❌ Chyba při stahování modelu: {e}")
        raise

# Stáhneme model před startem aplikace
download_model()

# Načtení modelu
try:
    model = load_model(LOCAL_MODEL_PATH, compile=False)
    logging.info(f"✅ Model úspěšně načten z {LOCAL_MODEL_PATH}")
except Exception as e:
    logging.error(f"❌ Chyba při načítání modelu: {e}")
    raise

# Inicializace aplikace Flask
app = Flask(__name__)
# Inicializace Google Cloud Error Reporting
client = error_reporting.Client()

@app.errorhandler(Exception)
def handle_exception(e):
    """Zachytí všechny chyby v aplikaci a pošle je do Google Cloud Error Reporting"""
    client.report_exception()  # Nahlásí chybu do Google Cloud
    return jsonify({"error": str(e)}), 500
app.config["PROPAGATE_EXCEPTIONS"] = True  # Logování detailních chyb

@app.route("/debug-env")
def debug_env():
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"),
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "NOT SET"),
    }

@app.route('/check-files')
def check_files():
    import os
    files = os.listdir(os.getcwd())  # Vypíše soubory v aktuální složce
    return jsonify({'files': files})

# Funkce pro získání aktuální ceny XRP z Binance API
def get_current_xrp_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT"
        response = requests.get(url)
        data = response.json()

        logging.debug(f"Binance API response: {data}")

        if "price" in data:
            current_price = float(data["price"])
            logging.info(f"✅ Current XRP price fetched: {current_price}")
            return current_price
        else:
            logging.error(f"❌ API response does not contain 'price' key! Response: {data}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error fetching XRP price: {e}")
        return None
    except ValueError as e:
        logging.error(f"❌ Invalid data format: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        return None

# Endpoint pro automatickou predikci
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing JSON in request'}), 400

        api_key = data.get('api_key')
        if api_key not in VALID_API_KEYS:
            return jsonify({'error': 'Invalid API key'}), 401

        current_price = get_current_xrp_price()
        if current_price is None:
            return jsonify({'error': 'Could not fetch current XRP price'}), 500

        X = np.array([[current_price, current_price]])
        prediction = model.predict(X)
        predicted_price = float(prediction[0][0])

        return jsonify({'current_price': current_price, 'predicted_price': predicted_price})

    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"❌ CHYBA: {error_trace}")
        return jsonify({'error': str(e), 'traceback': error_trace}), 500

# Definování cesty k šabloně
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logging.info(f"🚀 Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)