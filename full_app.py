import os
import logging
import traceback
import platform
# from flask import Flask, render_template, request, jsonify
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
import requests
from google.cloud import storage
import tensorflow as tf

# Primární import; pokud selže, použijeme alternativní cestu.
try:
    from tensorflow.keras.models import Model, load_model
except ImportError:
    from tensorflow.python.keras.models import Model, load_model  # type: ignore

# Potlačíme nepodstatné logy TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Použijeme pouze CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    tf.config.set_visible_devices([], "GPU")
    logging.info("GPU disabled; using CPU.")
except Exception as e:
    logging.warning(f"Error disabling GPU: {e}")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Nastavení cesty pro uložení modelu:
if platform.system() == "Windows":
    LOCAL_MODEL_PATH = "xrp_model.h5"
else:
    LOCAL_MODEL_PATH = "/tmp/xrp_model.h5"

# Konfigurace Google Cloud Storage
BUCKET_NAME = "xrp-model-storage"
MODEL_FILE = "xrp_model.h5"

def download_model():
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        logging.info(f"Používám bucket: {bucket.name}")
        blob = bucket.blob(MODEL_FILE)
        if not blob.exists():
            logging.error(f"Blob '{MODEL_FILE}' neexistuje v bucketu '{BUCKET_NAME}'!")
        else:
            logging.info(f"Blob '{MODEL_FILE}' nalezen, velikost: {blob.size} bytů")
        blob.download_to_filename(LOCAL_MODEL_PATH)
        logging.info(f"Model stažen do {LOCAL_MODEL_PATH}")
        if os.path.exists(LOCAL_MODEL_PATH):
            logging.info(f"Soubor {LOCAL_MODEL_PATH} byl úspěšně stažen.")
        else:
            logging.error(f"Soubor {LOCAL_MODEL_PATH} nebyl nalezen po stažení!")
        logging.info("Obsah /tmp: " + str(os.listdir("/tmp")))
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        logging.error("download_model() selhalo, pokračuji bez modelu.")
# Voláme download_model() – ujistěte se, že váš služební účet má roli "Storage Object Viewer"
download_model()

# Načtení modelu s explicitní typovou anotací; pokud dojde k chybě, nastavíme model na None
try:
    model: Model = load_model(LOCAL_MODEL_PATH, compile=False)  # type: ignore
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Import API klíčů ze souboru VALID_API_KEYS.py (soubor musí být ve stejném adresáři)
from VALID_API_KEYS import VALID_API_KEYS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "moje_tajne_heslo")

# @app.route('/')
# def index():
#     return render_template('index.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        # Získáme API klíč z formuláře
        key = request.form.get('api_key')
        # Načteme platné API klíče z prostředí nebo použijeme importované VALID_API_KEYS
        env_api_keys = os.getenv("VALID_API_KEYS")
        if env_api_keys:
            valid_api_keys = set(env_api_keys.split(","))
        else:
            valid_api_keys = VALID_API_KEYS

        if key in valid_api_keys:
            session['logged_in'] = True
            session['api_key'] = key
            return redirect(url_for('index'))
        else:
            error = "Neplatný API klíč"
    # Vždy zobrazíme stejnou šablonu (např. signin.html), která má přihlašovací formulář
    return render_template("signin.html", error=error)

@app.route('/')
def index():
    # Příklad hlavní stránky
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route('/debug-env')
def debug_env():
    files = os.listdir(".")
    return jsonify({
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "NOT SET"),
        "VALID_API_KEYS_FROM_FILE": list(VALID_API_KEYS),
        "files_in_root": files
    })

def get_current_xrp_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ripple&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        logging.info(f"CoinGecko API response status: {response.status_code}")
        logging.info(f"CoinGecko API response text: {response.text}")
        if response.status_code != 200:
            logging.error(f"Non-200 response from CoinGecko API: {response.text}")
            return None
        data = response.json()
        if "ripple" in data and "usd" in data["ripple"]:
            price = data["ripple"]["usd"]
            logging.info(f"Fetched XRP price from CoinGecko: {price}")
            return float(price)
        else:
            logging.error("Expected fields not found in CoinGecko API response")
            return None
    except Exception as e:
        logging.error(f"Error fetching XRP price: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "api_key" not in data:
            return jsonify({"error": "Missing API key"}), 400

        env_api_keys = os.getenv("VALID_API_KEYS")
        if env_api_keys:
            valid_api_keys = set(env_api_keys.split(","))
        else:
            valid_api_keys = VALID_API_KEYS

        if data["api_key"] not in valid_api_keys:
            return jsonify({"error": "Invalid API key"}), 401

        current_price = get_current_xrp_price()
        if current_price is None:
            return jsonify({"error": "Failed to fetch XRP price"}), 500

        X = np.array([[current_price, current_price]])
        if model is None:
            predicted_price = 0.0
        else:
            prediction = model.predict(X)
            predicted_price = float(prediction[0][0])
        return jsonify({"current_price": current_price, "predicted_price": predicted_price})
    except Exception as e:
        logging.error(f"Error in /predict: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/list-tmp')
def list_tmp():
    try:
        files = os.listdir("/tmp")
        return jsonify({"files_in_tmp": files})
    except Exception as e:
        return jsonify({"error": str(e)})

from werkzeug.exceptions import HTTPException
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    test_price = get_current_xrp_price()
    logging.info(f"Testovací cena XRP: {test_price}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))