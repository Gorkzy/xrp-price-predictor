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
import json

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

# Nastavení velikosti vstupního okna pro krátkodobý a dlouhodobý model
WINDOW_SIZE_SHORT = 5    # Krátkodobý model
WINDOW_SIZE_LONG = 10    # Dlouhodobý model

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

# Stáhni modelové soubory z GCS
# Krátkodobý model
download_model("xrp_model_short.h5", "xrp_model_short.h5")
download_model("scaler_X_short.pkl", "scaler_X_short.pkl")
download_model("scaler_y_short.pkl", "scaler_y_short.pkl")
# Dlouhodobý model
download_model("xrp_model_long.h5", "xrp_model_long.h5")
download_model("scaler_X_long.pkl", "scaler_X_long.pkl")
download_model("scaler_y_long.pkl", "scaler_y_long.pkl")
# ARIMA model
download_model("xrp_model_arima.pkl", "xrp_model_arima.pkl")
download_model("scaler_X_arima.pkl", "scaler_X_arima.pkl")
download_model("scaler_y_arima.pkl", "scaler_y_arima.pkl")
# LSTM model
download_model("xrp_model_lstm.h5", "xrp_model_lstm.h5")
download_model("scaler_X_lstm.pkl", "scaler_X_lstm.pkl")
download_model("scaler_y_lstm.pkl", "scaler_y_lstm.pkl")

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

# Načtení ARIMA modelu a scalerů
try:
    model_arima = joblib.load("xrp_model_arima.pkl")
    logging.info("ARIMA model loaded successfully")
except Exception as e:
    logging.error(f"Error loading ARIMA model: {e}")
    model_arima = None

try:
    scaler_X_arima = joblib.load("scaler_X_arima.pkl")
    scaler_y_arima = joblib.load("scaler_y_arima.pkl")
    logging.info("ARIMA scalers loaded successfully")
except Exception as e:
    logging.error(f"Error loading ARIMA scalers: {e}")
    scaler_X_arima = None
    scaler_y_arima = None

# Načtení LSTM modelu a scalerů
try:
    model_lstm = load_model("xrp_model_lstm.h5", compile=False)
    logging.info("LSTM model loaded successfully")
except Exception as e:
    logging.error(f"Error loading LSTM model: {e}")
    model_lstm = None

try:
    scaler_X_lstm = joblib.load("scaler_X_lstm.pkl")
    scaler_y_lstm = joblib.load("scaler_y_lstm.pkl")
    logging.info("LSTM scalers loaded successfully")
except Exception as e:
    logging.error(f"Error loading LSTM scalers: {e}")
    scaler_X_lstm = None
    scaler_y_lstm = None

# Import API klíčů ze souboru VALID_API_KEYS.py
from VALID_API_KEYS import VALID_API_KEYS

app = Flask(__name__, static_folder='.')
app.secret_key = os.environ.get("SECRET_KEY", "moje_tajne_heslo")

# ---------------------- Přihlašovací funkce ----------------------
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('signin'))
    return render_template('index.html', api_key=session.get('api_key'))

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    error = None
    if request.method == 'POST':
        key = request.form.get('api_key')
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
    return render_template('signin.html', error=error)

# ---------------------- Funkce pro získání aktuální ceny ----------------------
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

# ---------------------- Ensemble Prediction Function ----------------------
def ensemble_prediction(preds, weights):
    """
    preds: dictionary s předpověďmi jednotlivých modelů, např.
           { "short": 2.67, "long": 2.71, "arima": 2.65, "lstm": 2.69 }
    weights: dictionary s vahami, např.
           { "short": 0.25, "long": 0.25, "arima": 0.25, "lstm": 0.25 }
    """
    total = 0
    total_weight = 0
    for key, pred in preds.items():
        if pred is not None:
            w = weights.get(key, 0)
            total += w * pred
            total_weight += w
    return total / total_weight if total_weight > 0 else None

# ---------------------- Predikční endpoint ----------------------
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

        # Výběr modelu dle parametru "model_type" – zde použijeme ensemble všech modelů
        current_price = get_current_xrp_price()
        if current_price is None:
            return jsonify({"error": "Failed to fetch XRP price"}), 500
        current_price = round(current_price, 5)

        # Vytvoříme vstupní data pro každý model: opakujeme aktuální cenu "window_size" krát
        X_input_short = np.array([[current_price] * WINDOW_SIZE_SHORT])
        X_input_long = np.array([[current_price] * WINDOW_SIZE_LONG])
        X_input_arima = X_input_short.copy()  # Pro ARIMA můžeme použít stejné okno jako pro short
        X_input_lstm = X_input_long.copy()    # Pro LSTM použijeme okno jako long

        preds = {}

        # Krátkodobý model
        if model_short and scaler_X_short and scaler_y_short:
            X_scaled_short = scaler_X_short.transform(X_input_short)
            pred_short = model_short.predict(X_scaled_short)
            pred_short = scaler_y_short.inverse_transform(pred_short)
            preds["short"] = round(float(pred_short[0][0]), 5)
        else:
            preds["short"] = None

        # Dlouhodobý model
        if model_long and scaler_X_long and scaler_y_long:
            X_scaled_long = scaler_X_long.transform(X_input_long)
            pred_long = model_long.predict(X_scaled_long)
            pred_long = scaler_y_long.inverse_transform(pred_long)
            preds["long"] = round(float(pred_long[0][0]), 5)
        else:
            preds["long"] = None

        # ARIMA model
        if model_arima and scaler_X_arima and scaler_y_arima:
            X_scaled_arima = scaler_X_arima.transform(X_input_arima)
            pred_arima = model_arima.predict(X_scaled_arima)
            pred_arima = scaler_y_arima.inverse_transform(pred_arima)
            preds["arima"] = round(float(pred_arima[0][0]), 5)
        else:
            preds["arima"] = None

        # LSTM model
        if model_lstm and scaler_X_lstm and scaler_y_lstm:
            X_scaled_lstm = scaler_X_lstm.transform(X_input_lstm)
            pred_lstm = model_lstm.predict(X_scaled_lstm)
            pred_lstm = scaler_y_lstm.inverse_transform(pred_lstm)
            preds["lstm"] = round(float(pred_lstm[0][0]), 5)
        else:
            preds["lstm"] = None

        # Pro jednoduchost nastavíme váhy – ty lze upravit podle validace
        weights = {"short": 0.25, "long": 0.25, "arima": 0.25, "lstm": 0.25}
        ensemble_pred = ensemble_prediction(preds, weights)
        if ensemble_pred is None:
            return jsonify({"error": "Ensemble prediction failed due to missing model outputs"}), 500

        return jsonify({
            "current_price": current_price,
            "predicted_price": ensemble_pred,
            "predictions": preds
            "model_type": "ensemble"
        })

    except Exception as e:
        logging.error(f"Error in /predict: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/latest_predictions')
def latest_predictions():
    try:
        with open("latest_predictions.json", "r") as f:
            predictions = json.load(f)
        return jsonify(predictions)
    except Exception as e:
        logging.error(f"Error reading latest predictions: {e}")
        return jsonify({"error": str(e)}), 500

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


# import os
# import logging
# import traceback
# import platform
# from tensorflow.keras.models import load_model
# from flask import Flask, render_template, request, jsonify, redirect, url_for, session
# import numpy as np
# import requests
# import json
# from google.cloud import storage
# import tensorflow as tf
# import joblib  # Pro načtení scalerů

# # Potlačíme nepodstatné logy TensorFlow
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # Použijeme pouze CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# try:
#     tf.config.set_visible_devices([], "GPU")
#     logging.info("GPU disabled; using CPU.")
# except Exception as e:
#     logging.warning(f"Error disabling GPU: {e}")

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Nastavení velikosti vstupního okna pro modely
# WINDOW_SIZE_SHORT = 5   # Krátkodobý model
# WINDOW_SIZE_LONG = 10   # Dlouhodobý model

# # Konfigurace Google Cloud Storage
# BUCKET_NAME = "xrp-model-storage"

# def download_model(model_file, local_path):
#     try:
#         client = storage.Client()
#         bucket = client.bucket(BUCKET_NAME)
#         logging.info(f"Using bucket: {bucket.name}")
#         blob = bucket.blob(model_file)
#         if not blob.exists():
#             logging.error(f"Blob '{model_file}' does not exist in bucket '{BUCKET_NAME}'!")
#         else:
#             logging.info(f"Blob '{model_file}' found, size: {blob.size} bytes")
#         blob.download_to_filename(local_path)
#         logging.info(f"Model '{model_file}' downloaded to {local_path}")
#         if os.path.exists(local_path):
#             logging.info(f"File {local_path} downloaded successfully.")
#         else:
#             logging.error(f"File {local_path} not found after download!")
#     except Exception as e:
#         logging.error(f"Error downloading model {model_file}: {e}")
#         logging.error("download_model() failed, proceeding without model.")

# # Stáhneme modely
# download_model("xrp_model_short.h5", "xrp_model_short.h5")
# download_model("xrp_model_long.h5", "xrp_model_long.h5")

# # Načtení krátkodobého modelu a scalerů
# try:
#     model_short = load_model("xrp_model_short.h5", compile=False)
#     logging.info("Short-term model loaded successfully")
# except Exception as e:
#     logging.error(f"Error loading short-term model: {e}")
#     model_short = None

# try:
#     scaler_X_short = joblib.load("scaler_X_short.pkl")
#     scaler_y_short = joblib.load("scaler_y_short.pkl")
#     logging.info("Short-term scalers loaded successfully")
# except Exception as e:
#     logging.error(f"Error loading short-term scalers: {e}")
#     scaler_X_short = None
#     scaler_y_short = None

# # Načtení dlouhodobého modelu a scalerů
# try:
#     model_long = load_model("xrp_model_long.h5", compile=False)
#     logging.info("Long-term model loaded successfully")
# except Exception as e:
#     logging.error(f"Error loading long-term model: {e}")
#     model_long = None

# try:
#     scaler_X_long = joblib.load("scaler_X_long.pkl")
#     scaler_y_long = joblib.load("scaler_y_long.pkl")
#     logging.info("Long-term scalers loaded successfully")
# except Exception as e:
#     logging.error(f"Error loading long-term scalers: {e}")
#     scaler_X_long = None
#     scaler_y_long = None

# # Import API klíčů ze souboru VALID_API_KEYS.py
# from VALID_API_KEYS import VALID_API_KEYS

# app = Flask(__name__, static_folder='.')
# app.secret_key = os.environ.get("SECRET_KEY", "moje_tajne_heslo")

# # ---------------------- Přihlašovací funkce ----------------------

# @app.route('/')
# def index():
#     if not session.get('logged_in'):
#         return redirect(url_for('signin'))
#     # Předáme API klíč do šablony (uživatel je přihlášen)
#     return render_template('index.html', api_key=session.get('api_key'))

# @app.route('/signin', methods=['GET', 'POST'])
# def signin():
#     error = None
#     if request.method == 'POST':
#         key = request.form.get('api_key')
#         env_api_keys = os.getenv("VALID_API_KEYS")
#         if env_api_keys:
#             valid_api_keys = set(env_api_keys.split(","))
#         else:
#             valid_api_keys = VALID_API_KEYS
#         if key in valid_api_keys:
#             session['logged_in'] = True
#             session['api_key'] = key
#             return redirect(url_for('index'))
#         else:
#             error = "Neplatný API klíč"
#     return render_template('signin.html', error=error)

# # ---------------------- Funkce pro získání aktuální ceny ----------------------

# def get_current_xrp_price():
#     try:
#         url = "https://api.coingecko.com/api/v3/simple/price?ids=ripple&vs_currencies=usd"
#         response = requests.get(url, timeout=10)
#         logging.info(f"CoinGecko API response status: {response.status_code}")
#         logging.info(f"CoinGecko API response text: {response.text}")
#         if response.status_code != 200:
#             logging.error(f"Non-200 response from CoinGecko API: {response.text}")
#             return None
#         data = response.json()
#         if "ripple" in data and "usd" in data["ripple"]:
#             price = data["ripple"]["usd"]
#             logging.info(f"Fetched XRP price from CoinGecko: {price}")
#             return float(price)
#         else:
#             logging.error("Expected fields not found in CoinGecko API response")
#             return None
#     except Exception as e:
#         logging.error(f"Error fetching XRP price: {e}")
#         return None

# # ---------------------- Predikční endpoint ----------------------

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         if not data or "api_key" not in data:
#             return jsonify({"error": "Missing API key"}), 400

#         env_api_keys = os.getenv("VALID_API_KEYS")
#         if env_api_keys:
#             valid_api_keys = set(env_api_keys.split(","))
#         else:
#             valid_api_keys = VALID_API_KEYS

#         if data["api_key"] not in valid_api_keys:
#             return jsonify({"error": "Invalid API key"}), 401

#         # Vyber model podle parametru "model_type" (default "short")
#         model_type = data.get("model_type", "short").lower()
#         if model_type not in ["short", "long"]:
#             return jsonify({"error": "Invalid model_type. Must be 'short' or 'long'"}), 400

#         current_price = get_current_xrp_price()
#         if current_price is None:
#             return jsonify({"error": "Failed to fetch XRP price"}), 500

#         current_price = round(current_price, 5)

#         if model_type == "short":
#             if scaler_X_short is None or scaler_y_short is None:
#                 return jsonify({"error": "Short-term scalers not loaded"}), 500
#             model_used = model_short
#             scaler_X_used = scaler_X_short
#             scaler_y_used = scaler_y_short
#             window_size = WINDOW_SIZE_SHORT
#         else:
#             if scaler_X_long is None or scaler_y_long is None:
#                 return jsonify({"error": "Long-term scalers not loaded"}), 500
#             model_used = model_long
#             scaler_X_used = scaler_X_long
#             scaler_y_used = scaler_y_long
#             window_size = WINDOW_SIZE_LONG

#         # Vytvoříme vstupní data: opakujeme aktuální cenu window_size krát
#         X_input = np.array([[current_price] * window_size])
#         try:
#             X_scaled = scaler_X_used.transform(X_input)
#             logging.info(f"X_input: {X_input}")
#             logging.info(f"X_scaled: {X_scaled}")
#         except Exception as e:
#             logging.error(f"Error during normalization: {e}")
#             return jsonify({"error": "Error during normalization"}), 500

#         if model_used is None:
#             predicted_price = 0.0
#         else:
#             prediction = model_used.predict(X_scaled)
#             logging.info(f"Prediction before inverse transform: {prediction}")
#             try:
#                 predicted_price_arr = scaler_y_used.inverse_transform(prediction)
#                 predicted_price = round(float(predicted_price_arr[0][0]), 5)
#                 logging.info(f"Prediction after inverse transform: {predicted_price}")
#             except Exception as e:
#                 logging.error(f"Error during inverse transformation: {e}")
#                 return jsonify({"error": "Error during inverse transformation"}), 500

#         return jsonify({
#             "current_price": current_price,
#             "predicted_price": predicted_price,
#             "model_type": model_type
#         })

#     except Exception as e:
#         logging.error(f"Error in /predict: {e}\n{traceback.format_exc()}")
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# # ---------------------- Další endpoints ----------------------

# @app.route('/latest_predictions')
# def latest_predictions():
#     try:
#         with open("latest_predictions.json", "r") as f:
#             predictions = json.load(f)
#         return jsonify(predictions)
#     except Exception as e:
#         logging.error(f"Error reading latest predictions: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/list-tmp')
# def list_tmp():
#     try:
#         files = os.listdir("/tmp")
#         return jsonify({"files_in_tmp": files})
#     except Exception as e:
#         return jsonify({"error": str(e)})

# from werkzeug.exceptions import HTTPException
# @app.errorhandler(Exception)
# def handle_exception(e):
#     if isinstance(e, HTTPException):
#         return e
#     return jsonify({"error": "Internal Server Error"}), 500

# if __name__ == '__main__':
#     test_price = get_current_xrp_price()
#     logging.info(f"Test XRP price: {test_price}")
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))