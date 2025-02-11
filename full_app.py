import os
import logging
import traceback
import platform
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
import requests
from google.cloud import storage
import tensorflow as tf
import joblib  # Import joblib pro načtení scalerů
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler

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

# ... (definice /login a /index endpointů) ...

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

        current_price = round(current_price, 5)

        X = np.array([[current_price, current_price]])

        # !!! KLÍČOVÁ ZMĚNA: Normalizace dat !!!
        try:
            scaler_X = joblib.load("scaler_X.pkl")  # Načtení scaleru
            X_scaled = scaler_X.transform(X) # Normalizace

            print("Tvar X před normalizací:", X.shape)
            print("Hodnoty X před normalizací:", X)
            print("Tvar X po normalizaci:", X_scaled.shape)
            print("Hodnoty X po normalizaci:", X_scaled)

        except Exception as e:
            logging.error(f"Chyba při normalizaci dat: {e}")
            return jsonify({"error": "Chyba při normalizaci dat"}), 500


        if model is None:
            predicted_price = 0.0
        else:
            prediction = model.predict(X_scaled) # Použij normalizovaná data

            print("Predikce (před inverzní transformací):", prediction)

            # !!! KLÍČOVÁ ZMĚNA: Inverzní transformace !!!
            try:
                scaler_y = joblib.load("scaler_y.pkl")
                predicted_price = scaler_y.inverse_transform(prediction)
                predicted_price = round(float(predicted_price[0][0]), 5)  # Změna: přístup k prvnímu prvku predikce
                print("Predikce (po inverzní transformaci):", predicted_price)
            except Exception as e:
                logging.error(f"Chyba při inverzní transformaci: {e}")
                return jsonify({"error": "Chyba při inverzní transformaci"}), 500

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