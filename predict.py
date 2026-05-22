import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys

MODEL_PATH = "classifier_model.h5"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run train.py first to train the model.")
    sys.exit(1)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def predict_image(path):
    """
    Predict whether an X-ray image is normal or abnormal.
    
    Args:
        path (str): Path to the image file
    """
    # Check if file exists
    if not os.path.exists(path):
        print(f"Error: Image file '{path}' not found.")
        return
    
    try:
        # Load and preprocess image
        img = image.load_img(path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        confidence = prediction * 100

        # Display results
        if prediction > 0.5:
            print(f"Prediction: Abnormal (Confidence: {confidence:.2f}%)")
        else:
            print(f"Prediction: Normal (Confidence: {(100-confidence):.2f}%)")
            
    except Exception as e:
        print(f"Error processing image: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with your test image path
    test_image = "dataset/normal/normal(1).jpg"
    predict_image(test_image)
