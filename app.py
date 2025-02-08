from flask import Flask, render_template, request, jsonify 
import numpy as np 
import os 
import logging 
import requests 
from google.cloud import storage, error_reporting 
from dotenv import load_dotenv 
import traceback 
from tensorflow.keras.models import load_model  # type: ignore 

# 💾 Načti proměnné z .env souboru (pro lokální testování) 
load_dotenv() 

# 🚀 Vynucení použití CPU pro TensorFlow 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import tensorflow as tf 

try: 
    tf.config.set_visible_devices([], "GPU") 
    logging.info("✅ GPU zakázáno. Používám CPU.") 
except Exception as e: 
    logging.warning(f"⚠️ Problém při zakázání GPU: {e}") 

# 📝 Nastavení logování 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') 

# 🔑 Načtení API klíčů 
VALID_API_KEYS = set(os.getenv("VALID_API_KEYS", "").split(",")) 
logging.info(f"✅ Načtené API klíče: {VALID_API_KEYS}") 
if not VALID_API_KEYS: 
    logging.error("❌ Chyba: API klíče nebyly nalezeny v prostředí!") 

# ☁️ Google Cloud Storage nastavení 
BUCKET_NAME = "xrp-model-storage"  # Název bucketu 
MODEL_FILE = "xrp_model.h5" 
LOCAL_MODEL_PATH = "xrp_model.h5" 

# 📥 Stahování modelu 

def download_model(): 
    try: 
        client = storage.Client() 
        bucket = client.bucket(BUCKET_NAME) 
        blob = bucket.blob(MODEL_FILE) 
        blob.download_to_filename(LOCAL_MODEL_PATH) 
        logging.info(f"✅ Model úspěšně stažen do {LOCAL_MODEL_PATH}") 
    except Exception as e: 
        logging.error(f"❌ Chyba při stahování modelu: {e}") 
        raise 

download_model() 

# 📦 Načtení modelu 
try: 
    model = load_model(LOCAL_MODEL_PATH, compile=False) 
    logging.info("✅ Model úspěšně načten.") 
except Exception as e: 
    logging.error(f"❌ Chyba při načítání modelu: {e}") 
    raise 

# 🔥 Inicializace aplikace Flask 
app = Flask(__name__) 

# 🚨 Google Cloud Error Reporting 
error_client = error_reporting.Client() 

def report_error(e): 
    error_client.report_exception() 
    return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500 

@app.errorhandler(Exception) 
def handle_exception(e): 
    return report_error(e) 

# 🔍 Debugovací endpoint 
@app.route('/debug-env') 
def debug_env(): 
    return jsonify({ 
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "NOT SET"), 
        "VALID_API_KEYS": os.getenv("VALID_API_KEYS", "NOT SET") 
    }) 

# 📊 Získání aktuální ceny XRP 

def get_current_xrp_price(): 
    try: 
        url = "https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT" 
        response = requests.get(url) 
        data = response.json() 
        return float(data["price"]) if "price" in data else None 
    except Exception as e: 
        logging.error(f"❌ Chyba při získávání ceny XRP: {e}") 
        return None 

# 🔮 Predikční endpoint 
@app.route('/predict', methods=['POST']) 
def predict(): 
    try: 
        data = request.get_json() 
        if not data or "api_key" not in data: 
            return jsonify({"error": "Missing API key"}), 400 
        if data["api_key"] not in VALID_API_KEYS: 
            return jsonify({"error": "Invalid API key"}), 401 
        current_price = get_current_xrp_price() 
        if current_price is None: 
            return jsonify({"error": "Failed to fetch XRP price"}), 500 
        X = np.array([[current_price, current_price]]) 
        prediction = model.predict(X) 
        return jsonify({"current_price": current_price, "predicted_price": float(prediction[0][0])}) 
    except Exception as e: 
        return report_error(e) 

# 🌍 Hlavní stránka 
@app.route('/') 
def index(): 
    return render_template('index.html') 

# 🚀 Spuštění aplikace 
if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8080)
 