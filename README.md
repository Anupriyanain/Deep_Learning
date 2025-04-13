
# 🧠 Brain Tumor Detection using Deep Learning

This project leverages **deep learning models** to automatically detect brain tumors from MRI images. It includes **image preprocessing, feature engineering, data augmentation**, and three model architectures — from scratch CNN, ResNet50, and EfficientNetB3. The models were evaluated on real data and tested for predictive reliability using unseen MRI scans.

---

## 📌 Project Highlights

- 🔍 **EDA & Visualization:** Class distribution, sample images, image sizes, grayscale vs. colored checks.
- ⚙️ **Preprocessing:** Includes resizing, grayscale conversion, Otsu segmentation, and Local Binary Pattern (LBP) extraction.
- 🔄 **Data Augmentation:** Applied with `ImageDataGenerator` to increase model generalization.
- 🧠 **Models Implemented:**
  - **CNN**: Custom convolutional neural network from scratch.
  - **ResNet50**: Transfer learning with two-phase fine-tuning.
  - **EfficientNetB3**: State-of-the-art model with extensive preprocessing and tuning.
- 📈 **Evaluation:** Accuracy, loss curves, and test metrics.
- 🖼️ **Predictions on New Images**: Visual output with confidence score.

---

## 📁 Dataset Structure

Ensure your dataset is organized as follows:

```
/archive
├── no/
│   └── *.jpg (MRI scans without tumor)
└── yes/
    └── *.jpg (MRI scans with tumor)
```

---

## ⚙️ Environment Setup

Install the required dependencies:

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-image
```

---

## 🚀 How to Run

- Execute `deeplearning_project.py`:
  ```bash
  python deeplearning_project.py
  ```
- Or open the Colab notebook:
  ```
  DeepLearning_Project.ipynb
  ```

---

## 🧠 Models in Detail

### 1. 🧪 CNN (Custom)
- Input shape: `(128, 128, 3)` from gray, segmented, and LBP channels.
- Architecture:
  - 2 Conv2D + MaxPooling
  - Flatten + Dense + Dropout
- Output: Sigmoid for binary classification
- Regularization: L2 and Dropout (0.5)

### 2. 🔁 ResNet50
- Pretrained on ImageNet.
- Input shape: RGB 128x128
- Training:
  - Phase 1: Freeze base layers
  - Phase 2: Unfreeze and fine-tune last layers
- Optimized using Adam and EarlyStopping.

### 3. 💡 EfficientNetB3
- Input shape: `(300, 300, 3)`
- Pretrained backbone (ImageNet)
- Custom dense head
- Optimized with class weights for imbalance
- Final test accuracy displayed

---

## 📊 Evaluation & Metrics

- Plots of Training vs Validation Accuracy & Loss
- Final test set accuracy printed for each model
- Visual predictions with image + confidence score

Example:

```python
predict_with_efficientnet("/content/archive/yes/Y10.jpg")
```

---

## 🔍 Sample Preprocessing Pipeline

- Load image → Resize → Convert to gray
- Segment with Otsu → Extract LBP → Stack channels
- Augment using rotation, flip, zoom

---

## 🏁 Future Work

- Incorporate more classes (e.g., tumor type)
- Use attention mechanisms (e.g., ViT, Swin Transformer)
- Deploy as a web application or mobile inference app

---

## 📂 File Overview

| File | Description |
|------|-------------|
| `deeplearning_project.py` | End-to-end executable script |
| `DeepLearning_Project.ipynb` | Colab version with results & plots |
| `README.md` | Project documentation |

---

> 💡 **Note:** Ensure sufficient GPU resources when training EfficientNetB3. It performs best on machines with high memory and processing power.

