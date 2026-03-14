# 🌾 Agricultural Pest Identification using Deep Learning

A deep neural network model for classifying agricultural pests from images, built with TensorFlow and transfer learning on VGG16. Designed to assist farmers and agricultural researchers in identifying harmful insects in the field.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Classes](#classes)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

---

## Overview

This project implements an image classification pipeline to identify 12 categories of agricultural pests. It leverages **VGG16** (pre-trained on ImageNet) as a feature extractor with custom dense layers for fine-tuned classification. The goal is to enable fast and accurate pest identification to support timely intervention in agricultural settings.

accuracy: 0.9445 loss: 0.1753

---

## Dataset

The dataset is sourced from Kaggle:

> **[Agricultural Pests Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset)**

It can be downloaded directly using the `kagglehub` library:

```python
import kagglehub
path = kagglehub.dataset_download("vencerlanz09/agricultural-pests-image-dataset")
```

The dataset is **balanced across all 12 classes**, so no oversampling or undersampling was required.

---

## Classes

The model identifies the following 12 agricultural pests:

| # | Pest |
|---|------|
| 1 | Ants |
| 2 | Bees |
| 3 | Beetle |
| 4 | Caterpillar |
| 5 | Earthworm |
| 6 | Earwig |
| 7 | Grasshopper |
| 8 | Moth |
| 9 | Slug |
| 10 | Snail |
| 11 | Wasp |
| 12 | Weevil |

---

## Model Architecture

The model uses **transfer learning** on top of VGG16:

- **Base Model**: VGG16 (pretrained on ImageNet, `include_top=False`)
  - All layers frozen except the **last 4 layers**, which are fine-tuned
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dense(64, relu)`
  - `Dense(128, relu)`
  - `Dense(256, relu)` → `Dropout(0.5)`
  - `Dense(512, relu)` → `BatchNormalization` → `Dropout(0.5)`
  - `Dense(12, softmax)` — output layer for 12 classes

**Compiler settings:**
- Optimizer: `Adam`
- Loss: `sparse_categorical_crossentropy`
- Metric: `accuracy`

---

## Installation

1. **Clone the repository**
   ```bash
   git clone [ https://github.com/KenezNonwar/Agricultural_Insect_identification.git ]
   cd agricultural-pest-identification
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow==2.19.0 kagglehub opencv-python scikit-learn matplotlib pandas numpy
   ```

---

## Usage

### Running the Notebook

Open `Agriculture_NN_project.ipynb` in **Google Colab** (recommended, GPU: T4) or Jupyter Notebook:

```bash
jupyter notebook Agriculture_NN_project.ipynb
```

### Running Inference on a Single Image

```python
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('Argiculture_Model_v3.keras')
IMG_SIZE = 200

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

img = load_image('path/to/your/image.jpg')
prediction = model.predict(img)
print("Predicted class index:", np.argmax(prediction))
```

---

## Training

The model is trained with the following configuration:

- **Image size**: 200 × 200 pixels
- **Batch size**: 32
- **Train/Test split**: 80% / 20% (random state: 47)
- **Epochs**: up to 10
- **Callbacks**:
  - `EarlyStopping` (patience=5, restores best weights)
  - `ReduceLROnPlateau` (patience=3, factor=0.5)

The trained model is saved as `Argiculture_Model_v3.keras`.

Or Download it from Drive to work locally
https://drive.google.com/file/d/1kLqu3vqkt81-uLzAmsoqk68bJYwzcVNC/view?usp=sharing

---

## Results

The model was evaluated on the held-out test set. Sample predictions confirmed correct identification of pests such as grasshoppers and earthworms from unseen images.

> To view a full evaluation, run the **Testing** section in the notebook which randomly samples test images and displays actual vs. predicted labels.

---

## Project Structure

```
agricultural-pest-identification/
│
├── Agriculture_NN_project.ipynb   # Main notebook
├── Argiculture_Model_v3.keras     # Saved trained model
└── README.md                      # Project documentation
```

---

## Dependencies

| Library | Version |
|--------|---------|
| TensorFlow | 2.19.0 |
| Keras | ≥ 3.5.0 |
| NumPy | < 2.2.0 |
| OpenCV | latest |
| scikit-learn | latest |
| pandas | latest |
| matplotlib | latest |
| kagglehub | latest |
| netron | latest |

---

## 🚀 Run on Google Colab

This project was developed and tested on **Google Colab with a T4 GPU**. It is recommended to use a GPU runtime for training.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

For Visuals i used repo from
https://github.com/HarisIqbal88/PlotNeuralNet
---
## 📄 License

This project is open source and available under the [MIT License](LICENSE).
