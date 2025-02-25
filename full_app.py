import os
import logging
import traceback
import platform
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
import requests
from google.cloud import storage
import tensorflow as tf
import joblib  # Pro načtení scalerů

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

# Nastavíme velikost vstupního okna pro oba modely
WINDOW_SIZE_SHORT = 5   # Pro krátkodobý model
WINDOW_SIZE_LONG = 10   # Pro dlouhodobý model

# Konfigurace Google Cloud Storage
BUCKET_NAME = "xrp-model-storage"

def download_model(model_file, local_path):
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        logging.info(f"Using bucket: {bucket.name}")
        blob = bucket.blob(model_file)
        if not blob.exists():
            logging.error(f"Blob '{model_file}' does not exist in bucket '{BUCKET_NAME}'!")
        else:
            logging.info(f"Blob '{model_file}' found, size: {blob.size} bytes")
        blob.download_to_filename(local_path)
        logging.info(f"Model '{model_file}' downloaded to {local_path}")
        if os.path.exists(local_path):
            logging.info(f"File {local_path} downloaded successfully.")
        else:
            logging.error(f"File {local_path} not found after download!")
    except Exception as e:
        logging.error(f"Error downloading model {model_file}: {e}")
        logging.error("download_model() failed, proceeding without model.")

# Stáhneme oba modely z GCS
download_model("xrp_model_short.h5", "xrp_model_short.h5")
download_model("xrp_model_long.h5", "xrp_model_long.h5")

# Načtení krátkodobého modelu a scalerů
try:
    model_short = load_model("xrp_model_short.h5", compile=False)
    logging.info("Short-term model loaded successfully")
except Exception as e:
    logging.error(f"Error loading short-term model: {e}")
    model_short = None

try:
    scaler_X_short = joblib.load("scaler_X_short.pkl")
    scaler_y_short = joblib.load("scaler_y_short.pkl")
    logging.info("Short-term scalers loaded successfully")
except Exception as e:
    logging.error(f"Error loading short-term scalers: {e}")
    scaler_X_short = None
    scaler_y_short = None

# Načtení dlouhodobého modelu a scalerů
try:
    model_long = load_model("xrp_model_long.h5", compile=False)
    logging.info("Long-term model loaded successfully")
except Exception as e:
    logging.error(f"Error loading long-term model: {e}")
    model_long = None

try:
    scaler_X_long = joblib.load("scaler_X_long.pkl")
    scaler_y_long = joblib.load("scaler_y_long.pkl")
    logging.info("Long-term scalers loaded successfully")
except Exception as e:
    logging.error(f"Error loading long-term scalers: {e}")
    scaler_X_long = None
    scaler_y_long = None

# Import API klíčů ze souboru VALID_API_KEYS.py
from VALID_API_KEYS import VALID_API_KEYS

app = Flask(__name__, static_folder='.')
app.secret_key = os.environ.get("SECRET_KEY", "moje_tajne_heslo")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

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

        # Ověření API klíče
        env_api_keys = os.getenv("VALID_API_KEYS")
        if env_api_keys:
            valid_api_keys = set(env_api_keys.split(","))
        else:
            valid_api_keys = VALID_API_KEYS

        if data["api_key"] not in valid_api_keys:
            return jsonify({"error": "Invalid API key"}), 401

        # Výběr modelu dle parametru model_type ("short" nebo "long", výchozí je "short")
        model_type = data.get("model_type", "short").lower()
        if model_type not in ["short", "long"]:
            return jsonify({"error": "Invalid model_type. Must be 'short' or 'long'"}), 400

        current_price = get_current_xrp_price()
        if current_price is None:
            return jsonify({"error": "Failed to fetch XRP price"}), 500

        current_price = round(current_price, 5)

        # Nastavení proměnných dle model_type
        if model_type == "short":
            window_size = WINDOW_SIZE_SHORT
            if scaler_X_short is None or scaler_y_short is None:
                return jsonify({"error": "Short-term scalers not loaded"}), 500
            model_used = model_short
            scaler_X_used = scaler_X_short
            scaler_y_used = scaler_y_short
        else:  # long-term
            window_size = WINDOW_SIZE_LONG
            if scaler_X_long is None or scaler_y_long is None:
                return jsonify({"error": "Long-term scalers not loaded"}), 500
            model_used = model_long
            scaler_X_used = scaler_X_long
            scaler_y_used = scaler_y_long

        # Vytvoření vstupních dat: opakujeme aktuální cenu window_size krát
        X_input = np.array([[current_price] * window_size])

        # Normalizace vstupních dat
        try:
            X_scaled = scaler_X_used.transform(X_input)
            logging.info(f"X_input shape: {X_input.shape}")
            logging.info(f"X_input values: {X_input}")
            logging.info(f"X_scaled shape: {X_scaled.shape}")
            logging.info(f"X_scaled values: {X_scaled}")
        except Exception as e:
            logging.error(f"Error during normalization: {e}")
            return jsonify({"error": "Error during normalization"}), 500

        # Predikce
        if model_used is None:
            predicted_price = 0.0
        else:
            prediction = model_used.predict(X_scaled)
            logging.info(f"Prediction before inverse transform: {prediction}")
            try:
                predicted_price_arr = scaler_y_used.inverse_transform(prediction)
                predicted_price = round(float(predicted_price_arr[0][0]), 5)
                logging.info(f"Prediction after inverse transform: {predicted_price}")
            except Exception as e:
                logging.error(f"Error during inverse transformation: {e}")
                return jsonify({"error": "Error during inverse transformation"}), 500

        return jsonify({
            "current_price": current_price,
            "predicted_price": predicted_price,
            "model_type": model_type
        })

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
    logging.info(f"Test XRP price: {test_price}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))