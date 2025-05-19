import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model



# Load CSV data
csv_file = 'clinical_data.csv'
df = pd.read_csv(csv_file)

# Separate features and target variables
features = df[['age', 'sex', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']]
targets = df[['VA_2weeks', 'VA_3months', 'VA_6months', 'VA_12months']]
ids = df['id']  # This is used to load the corresponding images

# Normalize the tabular data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for later use

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(features, targets, ids, test_size=0.1, random_state=42)

# Function to load images corresponding to the CSV rows
def load_images(image_dir, ids):
    images = []
    for image_id in ids:
        image_path = os.path.join(image_dir, f'{image_id}.tiff')  # Assuming images are named as 0.tiff, 1.tiff, etc.
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image /= 255.0
        images.append(image)
    return np.array(images)

# Directory where images are stored
image_dir = 'octs'

# Load image data
train_images = load_images(image_dir, id_train)
test_images = load_images(image_dir, id_test)

# Define the model

# Image Input Branch
image_input = Input(shape=(224, 224, 3))
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
image_output = Dense(32, activation='relu')(x)

# Tabular Input Branch
tabular_input = Input(shape=(X_train.shape[1],))
y = Dense(128, activation='relu')(tabular_input)
y = Dense(64, activation='relu')(y)
tabular_output = Dense(32, activation='relu')(y)

# Combine the two branches
combined = Concatenate()([image_output, tabular_output])
z = Dense(64, activation='relu')(combined)
z = Dense(32, activation='relu')(z)
final_output = Dense(4, activation='linear')(z)  # 4 output units for the 4 target variables

# Create the model
model = Model(inputs=[image_input, tabular_input], outputs=final_output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Model summary
model.summary()

# Train the model
history = model.fit(
    [train_images, X_train],
    y_train,
    validation_data=([test_images, X_test], y_test),
    epochs=500,
    batch_size=8
)

# Save the model
model.save('multimodal_modelnew.h5')

# ===== EDITED PART BELOW =====

# Plot Loss Curves
# Save Loss and MAE Curves as images
plt.figure(figsize=(12, 5))

# Plot Loss Curves
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Plot MAE Curves
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('training_plots.png')

print("Plots saved as 'training_plots.png'. You can download and view it.")


# Save the model architecture as an image
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

print("Model architecture saved as 'model_architecture.png'.")

# ===== END OF EDITED PART =====


# # Import joblib for saving/loading the scaler

# # Load CSV data
# csv_file = 'clinical_data.csv'
# df = pd.read_csv(csv_file)

# # Separate features and target variables
# features = df[['age', 'sex', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_baseline']]
# targets = df[['VA_2weeks', 'VA_3months', 'VA_6months', 'VA_12months']]
# ids = df['id']  # This is used to load the corresponding images

# # Normalize the tabular data
# scaler = StandardScaler()
# features = scaler.fit_transform(features)

# # Save the scaler
# joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for later use

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(features, targets, ids, test_size=0.2, random_state=42)

# # Function to load images corresponding to the CSV rows
# def load_images(image_dir, ids):
#     images = []
#     for image_id in ids:
#         image_path = os.path.join(image_dir, f'{image_id}.tiff')  # Assuming images are named as 0.tiff, 1.tiff, etc.
#         image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#         image = tf.keras.preprocessing.image.img_to_array(image)
#         image /= 255.0
#         images.append(image)
#     return np.array(images)

# # Directory where images are stored
# image_dir = 'octs'

# # Load image data
# train_images = load_images(image_dir, id_train)
# test_images = load_images(image_dir, id_test)

# # Define the model

# # Image Input Branch
# image_input = Input(shape=(224, 224, 3))
# base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
# x = Flatten()(base_model.output)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# image_output = Dense(32, activation='relu')(x)

# # Tabular Input Branch
# tabular_input = Input(shape=(X_train.shape[1],))
# y = Dense(128, activation='relu')(tabular_input)
# y = Dense(64, activation='relu')(y)
# tabular_output = Dense(32, activation='relu')(y)

# # Combine the two branches
# combined = Concatenate()([image_output, tabular_output])
# z = Dense(64, activation='relu')(combined)
# z = Dense(32, activation='relu')(z)
# final_output = Dense(4, activation='linear')(z)  # 4 output units for the 4 target variables

# # Create the model
# model = Model(inputs=[image_input, tabular_input], outputs=final_output)

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Model summary
# model.summary()

# # Train the model
# history = model.fit(
#     [train_images, X_train],
#     y_train,
#     validation_data=([test_images, X_test], y_test),
#     epochs=10,
#     batch_size=10
# )

# # Save the model
# model.save('multimodal_model.h5')

# # To load the scaler during inference or testing, use the following code
# # scaler = joblib.load('scaler.pkl')  # Uncomment this line when you need to use the scaler
# # Import necessary libraries








