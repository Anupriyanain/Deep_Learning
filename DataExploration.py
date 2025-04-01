# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:09:58 2025

@author: amanm
"""

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
data_dir = unzip_dataset("C://Users/amanm/Downloads/archive (1).zip")
categories = ["no", "yes"]

# Count images in each category
category_counts = {category: len(os.listdir(os.path.join(data_dir, category))) for category in categories}
print("Dataset Summary:", category_counts)

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