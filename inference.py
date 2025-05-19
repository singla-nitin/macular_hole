import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load the trained model
model = load_model('multimodal_model.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Function to preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image /= 255.0
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to preprocess tabular data
def preprocess_features(features):
    features = np.array(features)
    features = scaler.transform(features)
    return features

# Function for making predictions
def make_prediction(image_path, features):
    # Preprocess image and features
    preprocessed_image = preprocess_image(image_path)
    preprocessed_features = preprocess_features([features])

    # Predict using the model
    prediction = model.predict([preprocessed_image, preprocessed_features])
    return prediction

# Example usage
# Image path and feature list
image_path = './octs/333.tiff' 
features = [68,1,0,11,1,298,61]  # Replace with the actual f

# Make prediction
prediction = make_prediction(image_path, features)
# print("Prediction:", prediction)


# # Display the predictions
# predicted_va_2weeks, predicted_va_3months, predicted_va_6months, predicted_va_12months = predictions[0]
print(f"Predicted VA at 2 weeks: {prediction[0][0]}")
print(f"Predicted VA at 3 months: {prediction[0][1]}")
print(f"Predicted VA at 6 months: {prediction[0][2]}")
print(f"Predicted VA at 12 months: {prediction[0][3]}")
