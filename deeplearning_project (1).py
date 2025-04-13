# -*- coding: utf-8 -*-
"""DeepLearning_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MnUY6CuPPxY41yzC9q6nxp6Bbq5pS43w
"""

!pip install tensorflow
!pip install opencv-python

import os
import zipfile
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
from skimage import feature
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Reshape, GlobalAveragePooling2D

# First, unzip the archive if it's a zip file
def unzip_dataset(zip_path):
    # Check if the path is a zip file
    if zip_path.endswith('.zip'):
        # Create a directory with the same name as the zip file (without .zip)
        extract_path = zip_path.replace('.zip', '')

        # Unzip the file if the extraction directory doesn't exist
        if not os.path.exists(extract_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

        return extract_path
    else:
        # If it's not a zip file, assume it's already an extracted directory
        return zip_path

# Update the data directory path
data_dir = unzip_dataset("/content/archive.zip")
categories = ["no", "yes"]

# Count images in each category
category_counts = {category: len(os.listdir(os.path.join(data_dir, category))) for category in categories}
print("Dataset Summary:", category_counts)

print("Path of unzipped data folder:", data_dir)

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()), palette="coolwarm")
plt.title("Class Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Images")
plt.show()

# Display sample images
def display_sample_images(category, num_samples=4):
    category_path = os.path.join(data_dir, category)
    images = os.listdir(category_path)[:num_samples]

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(category_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(category)
    plt.show()

# Show sample images from both categories
for category in categories:
    display_sample_images(category)

# Image size analysis
image_shapes = []
for category in categories:
    category_path = os.path.join(data_dir, category)
    images = os.listdir(category_path)[:50]  # Sampling 50 images per category
    for img_name in images:
        img = cv2.imread(os.path.join(category_path, img_name))
        if img is not None:
            image_shapes.append(img.shape[:2])  # Store (height, width)

# Convert to numpy array for analysis
image_shapes = np.array(image_shapes)

# Plot image size distribution
plt.figure(figsize=(8, 6))
sns.scatterplot(x=image_shapes[:, 1], y=image_shapes[:, 0], alpha=0.5)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution")
plt.show()

# Additional Insights
# 1. Checking unique image sizes
unique_sizes = Counter(map(tuple, image_shapes))
print("Unique Image Sizes:", unique_sizes)

# 2. Average Image Dimensions
avg_height = np.mean(image_shapes[:, 0])
avg_width = np.mean(image_shapes[:, 1])
print(f"Average Image Dimensions: {avg_width:.2f} x {avg_height:.2f}")

# 3. Checking grayscale vs. colored images
color_counts = {"Grayscale": 0, "Colored": 0}
for category in categories:
    category_path = os.path.join(data_dir, category)
    images = os.listdir(category_path)[:50]
    for img_name in images:
        img = cv2.imread(os.path.join(category_path, img_name))
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] == 3:
                color_counts["Grayscale"] += 1
            else:
                color_counts["Colored"] += 1

# Plot grayscale vs. colored distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=list(color_counts.keys()), y=list(color_counts.values()), palette="viridis")
plt.title("Grayscale vs. Colored Images")
plt.xlabel("Image Type")
plt.ylabel("Count")
plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
img_size = (128, 128)
categories = ["no", "yes"]
data_dir = "/content/archive"

# Feature extraction: Local Binary Pattern
def extract_lbp_features(gray_img):
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())  # Normalize
    return lbp

# Segmentation using Otsu Threshold
def segment_image(gray_img):
    thresh = threshold_otsu(gray_img)
    binary_mask = gray_img > thresh
    return binary_mask.astype(np.float32)


# Enhanced image loader with preprocessing
def load_and_preprocess_images(category):
     category_path = os.path.join(data_dir, category)
     images = []
     labels = []
     label = 0 if category == "no" else 1

     for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            gray = rgb2gray(img)
            segmented = segment_image(gray)
            lbp = extract_lbp_features(gray)

            # Stack original + segmented + LBP as channels
            processed_img = np.stack([gray, segmented, lbp], axis=-1)
            images.append(processed_img)
            labels.append(label)

     return np.array(images), np.array(labels)


# Load data with full preprocessing
x_no, y_no = load_and_preprocess_images("no")
x_yes, y_yes = load_and_preprocess_images("yes")

# Merge and split
X = np.concatenate((x_no, x_yes), axis=0)
y = np.concatenate((y_no, y_yes), axis=0)

from sklearn.model_selection import train_test_split

# First split off the training data (72%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.28, stratify=y, random_state=42)
# Then split remaining 28% into 20% test and 8% validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=20/28, stratify=y_temp, random_state=42)

# Data augmentation
augmentor = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the generator on training data
augmentor.fit(X_train)

# Show 5 preprocessed images
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[i][:,:,0], cmap='gray')
    plt.title("Preprocessed")
    plt.axis('off')
plt.suptitle("Sample Preprocessed Images")
plt.show()

# Show 5 augmented images
aug_iter = augmentor.flow(X_train, y_train, batch_size=5)
augmented_imgs, _ = next(aug_iter)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(augmented_imgs[i][:,:,0], cmap='gray')
    plt.title("Augmented")
    plt.axis('off')
plt.suptitle("Sample Augmented Images")
plt.show()

print(f"Preprocessed and augmented dataset loaded: {X.shape[0]} samples.")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

"""## **CNN Model**"""

#Feature enhancement
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
img_eq = cv2.equalizeHist(img_gray)  # Apply histogram equalization

#Edge detection
edges = cv2.Canny(img_gray, 50, 150)  # Use grayscale image for edge detection

from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    # Input layer
    keras.Input(shape=(128, 128, 3)),

    # Convolutional layers with regularization
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
    MaxPooling2D((2, 2)),

    # Flatten for dense layers
    Flatten(),

    # Dense layers with regularization
    Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Output layer
])

# --- Model Compilation and Training ---

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])


# --- Evaluation ---

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot Accuracy and Loss
plt.figure(figsize=(12, 4))  # Adjust figure size if needed

# Accuracy plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')  # Add test loss line
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu

def preprocess_image_for_cnn(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray = rgb2gray(img)

    # Segmentation (Otsu)
    thresh = threshold_otsu(gray)
    segmented = (gray > thresh).astype(np.float32)

    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())

    # Stack into 3 channels (same as training)
    stacked = np.stack([gray, segmented, lbp], axis=-1)
    stacked = np.expand_dims(stacked, axis=0)  # Add batch dimension
    return stacked

def predict_uploaded_image(img_path):
    processed_img = preprocess_image_for_cnn(img_path)
    prediction = model.predict(processed_img)[0][0]
    label = "🧠 Tumor Detected" if prediction > 0.5 else "✅ No Tumor Detected"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show image and result
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"{label} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

    return label, confidence

# --- Run this to test your model on a new image ---
# Replace with the correct file path
# test_image_path = "/content/archive/yes/Y101.jpg"
test_image_path = "/content/archive/no/12 no.jpg"
predict_uploaded_image(test_image_path)

"""## **RESNET50**"""

from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load ResNet50 and Add Custom Layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

num_classes = 2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.02))(x)  # Increased L2 regularization
x = Dropout(0.7)(x)  # Increased dropout rate
predictions = Dense(num_classes, activation='softmax')(x)

supervised_model = Model(inputs=base_model.input, outputs=predictions)

# 2. Freeze and Fine-tune (Gradual Unfreezing)
for layer in base_model.layers[:175]:  # Freeze more layers initially
    layer.trainable = False

# 3. Compile and Train with Early Stopping and Regularization (Stage 1)
supervised_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',  # Changed from categorical_crossentropy
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = supervised_model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# 4. Unfreeze More Layers and Train (Stage 2)
for layer in base_model.layers[140:]:  # Corrected unfreezing range
    layer.trainable = True

supervised_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Even lower learning rate
    loss='sparse_categorical_crossentropy',  # Keep consistent with stage 1
    metrics=['accuracy']
)

history = supervised_model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# 5. Evaluate the Model
test_loss, test_accuracy = supervised_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 6. Data Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# 7. Train with Augmented Data
history = supervised_model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Evaluate the model on the test data
test_loss, test_accuracy = supervised_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss after Data Augmentation: {test_loss:.4f}")
print(f"Test Accuracy after Data Augmentation: {test_accuracy:.4f}")

# Plot Accuracy and Loss
plt.figure(figsize=(12, 4))  # Adjust figure size if needed

# Accuracy plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')  # Add test loss line
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

import cv2
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# --- Prediction Helper Functions ---

def preprocess_image_for_prediction(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray = rgb2gray(img)

    # Segmentation
    thresh = threshold_otsu(gray)
    segmented = (gray > thresh).astype(np.float32)

    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())  # Normalize

    # Stack as 3 channels
    stacked = np.stack([gray, segmented, lbp], axis=-1)

    # Expand dims for batch shape
    stacked = np.expand_dims(stacked, axis=0)
    return stacked

def predict_brain_tumor(img_path):
    processed_img = preprocess_image_for_prediction(img_path)
    prediction = supervised_model.predict(processed_img)
    class_idx = np.argmax(prediction)
    label = "No Tumor" if class_idx == 0 else "Tumor Detected"

    # Display
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {label}")
    plt.axis("off")
    plt.show()

    return label

# --- Run Prediction ---
# Replace with your actual image path
# uploaded_image_path = "/content/brainmriimage.png"
# uploaded_image_path = "/content/archive/no/10 no.jpg"
uploaded_image_path ="/content/archive/yes/Y102.jpg"
predict_brain_tumor(uploaded_image_path)

"""## **EfficientNetB3**"""

# Step 2: Define paths for categories
data_dir = "/content/archive"

no_category_path = os.path.join(data_dir, 'no')
yes_category_path = os.path.join(data_dir, 'yes')

# Step 3: Feature extraction (LBP) and segmentation using Otsu Threshold
def extract_lbp_features(gray_img):
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())  # Normalize
    return lbp

def segment_image(gray_img):
    thresh = threshold_otsu(gray_img)
    binary_mask = gray_img > thresh
    return binary_mask.astype(np.float32)

# Step 4: Image Preprocessing (RGB-only)
def load_and_preprocess_images_rgb(category_path, label):
    images = []
    labels = []

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (300, 300))  # Resize to EfficientNetB3 input size
            processed_img = img
            images.append(processed_img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Step 5: Load data for both categories
x_no, y_no = load_and_preprocess_images_rgb(no_category_path, label=0)  # "No" category
x_yes, y_yes = load_and_preprocess_images_rgb(yes_category_path, label=1)  # "Yes" category

# Step 6: Merge both categories into one dataset
X = np.concatenate((x_no, x_yes), axis=0)
y = np.concatenate((y_no, y_yes), axis=0)

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

# 7: Stratified Shuffle Split (72% train, 8% validation, 20% test)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, temp_index in sss.split(X, y):
    X_train, X_temp = X[train_index], X[temp_index]
    y_train, y_temp = y[train_index], y[temp_index]

sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_index, test_index in sss_val.split(X_temp, y_temp):
    X_val, X_test = X_temp[val_index], X_temp[test_index]
    y_val, y_test = y_temp[val_index], y_temp[test_index]

# Check class distribution in splits
print(f"Train class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Validation class distribution:\n{pd.Series(y_val).value_counts()}")
print(f"Test class distribution:\n{pd.Series(y_test).value_counts()}")

from sklearn.utils import class_weight

# Step 8: Compute class weights to address class imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Check the computed class weights
print("Class Weights:", class_weights)

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB3

# Step 9: Define EfficientNetB3 Model
input_layer = Input(shape=(300, 300, 3))  # Input shape for RGB images
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=input_layer)
base_model.trainable = False  # Freeze the base model initially

# Add custom classifier on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=input_layer, outputs=predictions)

model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Step 10: Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 11: Train the model (first stage: freeze the base model)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weights,  # Add class weights
    callbacks=[early_stopping]
)

# Step 13: Plot loss and accuracy curves for training and validation
def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)  # Plot loss curve for the first training phase
plot_accuracy(history)  # Plot accuracy curve for the first training phase

# Step 14: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"EfficientNetB3 Test Accuracy: {test_accuracy:.4f}")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Prediction Helper Functions ---

def preprocess_image_for_efficientnet(img_path):
    """
    Preprocess the image for EfficientNetB3 model.
    - Resize to 300x300 (EfficientNetB3 input size).
    - Normalize using preprocess_input from EfficientNet.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = preprocess_input(img)  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_with_efficientnet(img_path):
    """
    Predict the class of the image using the EfficientNetB3 model.
    - Displays the image along with the prediction and confidence score.
    """
    processed_img = preprocess_image_for_efficientnet(img_path)
    prediction = model.predict(processed_img)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "No Tumor"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Display the image and prediction
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"{label} (Confidence: {confidence:.2f})")
    plt.axis("off")
    plt.show()

    return label

# --- Run Prediction ---
# Replace with your actual image path
# uploaded_image_path = "/content/archive/no/14 no.jpg"
uploaded_image_path = "/content/archive/yes/Y10.jpg"
predict_with_efficientnet(uploaded_image_path)