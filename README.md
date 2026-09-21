# Fashion Image Classification System (Fashion-MNIST)

An end-to-end Deep Learning application built with **TensorFlow / Keras** and **Streamlit** to classify fashion and clothing articles from the **Fashion-MNIST** dataset into 10 distinct categories.

---

## 📌 Project Overview

- **Dataset**: Fashion-MNIST (60,000 training images, 10,000 test images, 28x28 grayscale)
- **Framework**: TensorFlow 2.x / Keras 3
- **Model Architecture**: Deep Convolutional Neural Network (CNN) with 3 Conv-BatchNorm-Pool blocks and Dense Classification Head
- **Regularization Techniques**:
  - **Batch Normalization** after each convolutional and dense layer to reduce internal covariate shift
  - **Dropout** (20%, 25%, 30%, 40%) to eliminate feature co-adaptation
  - **L2 Weight Regularization** ($10^{-4}$) to constrain weight magnitudes
  - **Early Stopping & Learning Rate Scheduling** (`ReduceLROnPlateau`)
- **Web UI**: Interactive Streamlit dashboard with curated test set gallery, real-time prediction visualizer, Explainable AI (Grad-CAM), and diagnostic analytics.

---

## 🏷️ Categories (10 Classes)

| Index | Label | Icon | Description |
|---|---|:---:|---|
| 0 | T-shirt/top | 👕 | Short-sleeve casual tops, t-shirts, tees |
| 1 | Trouser | 👖 | Trousers, denim jeans, slacks |
| 2 | Pullover | 🧥 | Sweaters, knitwear, hoodies |
| 3 | Dress | 👗 | One-piece gowns and dresses |
| 4 | Coat | 🧥 | Overcoats, parkas, outerwear jackets |
| 5 | Sandal | 👡 | Open-toed summer footwear |
| 6 | Shirt | 👔 | Collared button-down shirts |
| 7 | Sneaker | 👟 | Athletic sports footwear |
| 8 | Bag | 👜 | Handbags, backpacks, totes |
| 9 | Ankle boot | 👢 | Ankle-length boots |

---

## 🚀 Quick Start

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Model Training & Evaluation
To retrain the model and regenerate plots/metrics:
```bash
python -m src.train
```

To evaluate an existing saved model:
```bash
python -m src.evaluate
```

### 3. Launch Streamlit Application
Double-click `run_app.bat` or run:
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
DL Proj/
├── model/
│   ├── fashion_cnn_model.keras    # Saved trained Keras model
│   ├── model_history.json         # Training & validation history
│   ├── test_metrics.json          # Precision, recall, F1, accuracy
│   └── sample_test_images.npz     # Cached sample test images
├── plots/
│   ├── training_curves.png        # Accuracy and Loss curves
│   ├── confusion_matrix.png       # Normalized confusion matrix heatmap
│   └── class_distribution.png     # Per-class F1-score chart
├── src/
│   ├── dataset.py                 # Dataset loader, normalization & split
│   ├── model.py                   # CNN architecture definition
│   ├── train.py                   # Training pipeline with callbacks
│   ├── evaluate.py                # Evaluation suite and plots
│   └── utils.py                   # Image preprocessing & inference helpers
├── app.py                         # Interactive Streamlit application
├── run_app.bat                    # One-click Windows launch script
├── requirements.txt               # Dependency specifications
└── README.md                      # Documentation
```
