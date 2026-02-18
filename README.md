# Sargassum Detection Model Suite

This repository contains a suite of machine learning and deep learning models trained to detect Sargassum in satellite imagery. It provides Python scripts for advanced users and detailed Jupyter Notebook tutorials for a step-by-step walkthrough of the entire research workflow, from training to classification.

## Project Overview

The goal of this project is to provide a robust and reproducible framework for Sargassum detection from multispectral satellite imagery. All models are designed for binary classification (Sargassum vs. No Sargassum) and utilize the same set of 5 input spectral bands.

*   **Input Features:** 5 spectral bands (Blue, Green, Red, NIR, SWIR1).
*   **Output:** A probabilistic fractional cover map indicating the likelihood of Sargassum presence in each pixel.

## Files and Directory Structure

```
.
├── README.md
├── requirements.txt
├── sargassum_data.csv
├── 1_train_models.py          # Train Keras MLP & 1D CNN (with TFLite quantization)
├── 1_train_models_ML.py       # Train XGBoost GPU model
├── 2_classify.py              # Inference with TFLite (INT8 / FLOAT16) models
├── 2_classify_ML.py           # Inference with XGBoost GPU model
├── Notebook_1_train_models.ipynb
├── Notebook_2_classify_sargassum.ipynb
├── satellite_data/
│   ├── landsat/
│   │   └── LC08_L1GT_016046_20150723_20200908_02_T2/
│   │       ├── *_B2.TIF, *_B3.TIF, *_B4.TIF, *_B5.TIF, *_B6.TIF
│   │       └── *_QA_PIXEL.TIF
│   └── sentinel/
│       └── S2B_MSIL2A_.../GRANULE/.../IMG_DATA/
│           ├── R10m/  (*_B02_10m.jp2, *_B03_10m.jp2, *_B04_10m.jp2, *_B08_10m.jp2)
│           └── R20m/  (*_B8A_20m.jp2, *_B11_20m.jp2, *_SCL_20m.jp2)
└── output/
    ├── fractional_cover_maps/     # GeoTIFF output maps from inference
    ├── model_cards/               # Suitability card images
    ├── plots_classification/      # Confusion matrix plots
    ├── tables_classification/     # Performance summary CSVs
    └── models_classification/
        ├── mlp_classifier_mixed.keras
        ├── mlp_classifier_int8.tflite
        ├── mlp_classifier_f16.tflite
        ├── cnn_classifier_mixed.keras
        ├── cnn_classifier_int8.tflite
        ├── cnn_classifier_f16.tflite
        ├── xgboost_gpu_classifier.joblib
        ├── scaler_classification.joblib
        └── label_encoder_classification.joblib
```

## Installation

1.  **Python Version:** This project was developed and tested using **Python 3.12.8**. It is recommended to use a virtual environment.

2.  **Install Dependencies:** Install all the required libraries using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

We provide two ways to use this repository: interactive Jupyter Notebooks (recommended) and the original Python scripts for automated workflows.

### Option A: Jupyter Notebook Tutorials (Recommended)

This is the easiest way to understand and replicate the workflow using the provided example data.

1.  **Train the Models:**
    *   Open and run the **`Notebook_1_train_models.ipynb`** notebook.
    *   This notebook will guide you through data loading, preprocessing, model training, and evaluation. It will generate all the model files and save them to `output/models_classification/`.

2.  **Classify Satellite Imagery:**
    *   Open and run the **`Notebook_2_classify_sargassum.ipynb`** notebook.
    *   This notebook shows how to load a pre-trained model and apply it to the example Landsat-8 and Sentinel-2 scenes included in the `satellite_data/` directory. It will generate and visualize the final fractional cover maps.

### Option B: Python Scripts (Advanced / Automated Use)

For automated or command-line workflows.

1.  **Train the Models:**
    *   Ensure `sargassum_data.csv` is present.
    *   For Keras MLP & 1D CNN models (with TFLite quantization):
    ```bash
    python 1_train_models.py
    ```
    *   For XGBoost GPU model:
    ```bash
    python 1_train_models_ML.py
    ```

2.  **Run Inference (Classification):**
    *   Place your satellite data in the `satellite_data/` directory.
    *   For TFLite (INT8/FLOAT16) models — edit the `MODELS_TO_RUN` list in `2_classify.py` to select models:
    ```bash
    python 2_classify.py
    ```
    *   For XGBoost GPU model:
    ```bash
    python 2_classify_ML.py
    ```

## Model Performance Summary

The following table summarizes the performance of each model on the held-out test set (30% of the data). Metrics are weighted-averaged across both 'Sargassum' and 'No Sargassum' classes.

| Model                   | Weighted F1-Score | Weighted Precision | Weighted Recall | Overall Accuracy | Training Time (s) |
| ----------------------- | ----------------- | ------------------ | --------------- | ---------------- | ----------------- |
| **Keras MLP (Mixed)**   | **0.9957**        | 0.9958             | 0.9958          | **0.9958**       | 79.87             |
| **1D CNN (Mixed)**      | 0.9953            | 0.9953             | 0.9953          | 0.9953           | 258.33            |
| **XGBoost (GPU)**       | 0.9728            | 0.9791             | 0.9703          | 0.9703           | **36.21**         |

> **Note:** All models use mixed precision (float16) training. TFLite INT8 and FLOAT16 quantized versions of the Keras models are also generated for edge deployment.


## Detailed Model Cards

Below are the specific hyperparameters and architectures for each model.

---

### 1. Keras MLP (Mixed Precision)

*   **Model Files:**
    *   `output/models_classification/mlp_classifier_mixed.keras` (FP16 Keras model)
    *   `output/models_classification/mlp_classifier_int8.tflite` (INT8 quantized)
    *   `output/models_classification/mlp_classifier_f16.tflite` (FLOAT16 quantized)
*   **Architecture:**
    *   Input(5,) → Flatten
    *   Dense(100, relu) → Dropout(0.4)
    *   Dense(50, relu) → Dropout(0.4)
    *   Dense(1, sigmoid)
    *   Optimizer: Adam (lr=0.001), Loss: binary_crossentropy
    *   Training: Mixed precision (float16), EarlyStopping (patience=15)
*   **Inference (TFLite):** Load `.tflite` file with `tf.lite.Interpreter`. Input shape: `(n, 5)`, scaled with `scaler_classification.joblib`.

---

### 2. 1D Convolutional Neural Network (1D CNN)

*   **Model Files:**
    *   `output/models_classification/cnn_classifier_mixed.keras` (FP16 Keras model)
    *   `output/models_classification/cnn_classifier_int8.tflite` (INT8 quantized)
    *   `output/models_classification/cnn_classifier_f16.tflite` (FLOAT16 quantized)
*   **Architecture:**
    *   Input(5, 1)
    *   Conv1D(32, k=3, relu, same) → MaxPool1D(2) → Dropout(0.3)
    *   Conv1D(64, k=3, relu, same) → MaxPool1D(2) → Dropout(0.4)
    *   Flatten → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)
    *   Optimizer: Adam (lr=0.001), Loss: binary_crossentropy
    *   Training: Mixed precision (float16), EarlyStopping (patience=15)
*   **Inference (TFLite):** Input shape: `(n, 5, 1)` — scaled data reshaped from `(n, 5)`.

---

### 3. XGBoost (GPU-Accelerated)

*   **Model File:** `output/models_classification/xgboost_gpu_classifier.joblib`
*   **Model Card:** `output/model_cards/suitability_card_xgboost.png`
*   **Best Parameters (via GridSearchCV, 5-fold StratifiedKFold):**
    *   `device`: 'cuda', `tree_method`: 'hist'
    *   `colsample_bytree`: 1.0
    *   `learning_rate`: 0.1
    *   `max_depth`: 5
    *   `n_estimators`: 150
    *   `subsample`: 1.0
    *   `scale_pos_weight`: auto-calculated from class distribution
*   **Inference:** Use `model.predict_proba(X_scaled)[:, positive_class_index]` for 'sargassum' probability.

---

## Citation

If you use these models or this suite in your research, please cite the associated paper and dataset:

> Echevarría-Rubio, J. M., Martínez-Flores, G., & Morales-Pérez, R. A. (2025). Sargassum Fractional Cover Estimation Models (Version 1.1) \[Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17246345