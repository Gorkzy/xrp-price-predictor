import tensorflow as tf

print(tf.__version__)

try:
    from tensorflow import keras
    print("Keras import successful!")
except ImportError as e:
    print(f"Keras import failed: {e}")