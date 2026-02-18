# --- Set Matplotlib Backend (MUST BE BEFORE importing pyplot or seaborn) ---
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for non-interactive plotting to file

# --- Standard Imports ---
import pandas as pd
import numpy as np
import os
import joblib # For saving sklearn models/scaler
import time # For timing operations
import traceback # For detailed error reporting
import matplotlib.pyplot as plt # Import AFTER setting backend
import seaborn as sns

# Scikit-learn imports (Only for utilities)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report)

# Configuration for Reproducibility
SEED = 42
np.random.seed(SEED) # For NumPy operations

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import Precision as KerasPrecision, Recall as KerasRecall
    from tensorflow.keras import mixed_precision
    # Use legacy Adam if needed
    try: AdamOptimizer = tf.keras.optimizers.Adam
    except AttributeError: AdamOptimizer = tf.keras.optimizers.legacy.Adam
    tf.random.set_seed(SEED) # For TensorFlow/Keras reproducibility

    # --- CONFIGURE GPU MEMORY GROWTH ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Set memory growth for: {gpus}")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")

    # --- ENABLE MIXED PRECISION ---
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"TensorFlow found and global mixed precision policy set to: {policy.name}")

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("CRITICAL: TensorFlow not found. This script requires TensorFlow to run.")

# --- Configuration ---
TEST_SIZE = 0.3
CLASS_COLUMN = 'class'

# Model Configuration
CNN_VALIDATION_SPLIT = 0.2
MLP_VALIDATION_SPLIT = 0.2
CNN_EPOCHS = 100
MLP_EPOCHS = 100
CNN_BATCH_SIZE = 64
MLP_BATCH_SIZE = 64
CNN_EARLY_STOPPING_PATIENCE = 15
MLP_EARLY_STOPPING_PATIENCE = 15


# --- Model Definitions & Hyperparameter Grids (CLASSIFICATION) ---

def build_mlp_classifier(input_shape):
    """Builds a Keras MLP model for BINARY CLASSIFICATION."""
    if not TF_AVAILABLE: return None

    model = Sequential([
        Input(shape=input_shape), # Input shape will be (num_features,)
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.4),
        Dense(50, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])

    optimizer = AdamOptimizer(learning_rate=0.001)
    metrics_to_compile = ['accuracy']
    if KerasPrecision and KerasRecall:
        metrics_to_compile.extend([KerasPrecision(name='precision'), KerasRecall(name='recall')])

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=metrics_to_compile)
    return model

def build_cnn_classifier(input_shape):
    """Builds the 1D CNN model for BINARY CLASSIFICATION."""
    if not TF_AVAILABLE: return None

    model = Sequential([
        Input(shape=input_shape), # Input shape will be (num_features, 1)
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    optimizer = AdamOptimizer(learning_rate=0.001)
    metrics_to_compile = ['accuracy']
    if KerasPrecision and KerasRecall:
        metrics_to_compile.extend([KerasPrecision(name='precision'), KerasRecall(name='recall')])

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=metrics_to_compile)
    return model

# --- Plotting Confusion Matrix ---
def plot_confusion_matrix(y_true, y_pred_labels, classes_display, model_name, output_dir):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_labels, labels=np.arange(len(classes_display)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_display)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
        ax.set_title(f'Confusion Matrix - {model_name}')
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('.', '')}.png")
        plt.savefig(output_path, dpi=300)
        print(f"CM saved: {output_path}")
    except Exception as e:
        print(f"Error plotting/saving CM for {model_name}: {e}")
    finally:
        plt.close(fig)

# --- Re-usable function to get an FP32 model with trained weights ---
def get_fp32_model_with_weights(keras_model_path):
    # 1. Load the trained mixed_precision model
    try:
        mixed_model = tf.keras.models.load_model(keras_model_path)
        trained_weights = mixed_model.get_weights()
    except Exception as e:
        print(f"Error loading trained model {keras_model_path}: {e}")
        return None

    # 2. Create an identical FP32 model
    mixed_precision.set_global_policy('float32') # Set policy

    try:
        if 'cnn' in keras_model_path:
            fp32_model = build_cnn_classifier(mixed_model.input_shape[1:])
        elif 'mlp' in keras_model_path:
            fp32_model = build_mlp_classifier(mixed_model.input_shape[1:])
        else:
            raise ValueError(f"Unknown model type for quantization: {keras_model_path}")

        fp32_model.set_weights(trained_weights)
        print("FP32 model created with trained weights.")
    except Exception as e:
        print(f"Error creating FP32 model structure: {e}")
        mixed_precision.set_global_policy('mixed_float16') # Reset policy
        return None

    # 3. Set policy back to mixed_float16 for any subsequent training
    mixed_precision.set_global_policy('mixed_float16')
    return fp32_model

# --- NEW: Quantization Function for INT8 ---
def quantize_model_int8(fp32_model, output_tflite_path, representative_dataset_gen):
    """Quantizes a given FP32 Keras model to INT8."""
    print(f"\n--- Starting INT8 Quantization -> {output_tflite_path} ---")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(fp32_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        tflite_int8_model = converter.convert()

        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_int8_model)

        print(f"Successfully quantized and saved INT8 model to: {output_tflite_path}")
        return True
    except Exception as e:
        print(f"Error during TFLite INT8 conversion: {e}")
        return False

# --- NEW: Quantization Function for FLOAT16 ---
def quantize_model_f16(fp32_model, output_tflite_path):
    """Quantizes a given FP32 Keras model to FLOAT16."""
    print(f"\n--- Starting FLOAT16 Quantization -> {output_tflite_path} ---")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(fp32_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # This is the key line for float16 quantization
        converter.target_spec.supported_types = [tf.float16]

        tflite_f16_model = converter.convert()

        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_f16_model)

        print(f"Successfully quantized and saved FLOAT16 model to: {output_tflite_path}")
        return True
    except Exception as e:
        print(f"Error during TFLite FLOAT16 conversion: {e}")
        return False


# --- Main Training & Evaluation Function (CLASSIFICATION) ---
def train_evaluate_classifiers(csv_path, output_dir_models, output_dir_plots, output_dir_tables, current_positive_class_label):
    if not TF_AVAILABLE:
        print("TensorFlow not found. Aborting training script.")
        return

    print(f"--- Starting CLASSIFICATION Model Training & Evaluation ---")
    print(f"Positive class: '{current_positive_class_label}'")
    print(f"Loading data from: {csv_path}")
    try: df = pd.read_csv(csv_path)
    except FileNotFoundError: print(f"Error: CSV not found: {csv_path}"); return
    except Exception as e: print(f"Error loading CSV: {e}"); return

    if CLASS_COLUMN not in df.columns:
        print(f"Error: Class column '{CLASS_COLUMN}' not found."); return

    feature_cols = [col for col in df.columns if col != CLASS_COLUMN]
    if not feature_cols:
        print("Error: No feature columns found."); return
    print(f"Using features: {list(feature_cols)}")
    X = df[feature_cols].values
    y_labels_str = df[CLASS_COLUMN].values

    try: X = X.astype(np.float64)
    except ValueError as e:
        print(f"Error: Could not convert feature columns to numeric. Error: {e}"); return

    le = LabelEncoder()
    sargassum_encoded_value = -1
    try:
        y = le.fit_transform(y_labels_str)
        class_names_str = le.classes_
        if len(class_names_str) != 2:
            print(f"Error: Expected 2 classes, found {len(class_names_str)}: {class_names_str}."); return
        print(f"Class mapping: {list(zip(class_names_str, le.transform(class_names_str)))}")

        try:
            sargassum_encoded_value = le.transform([current_positive_class_label])[0]
            print(f"The positive class '{current_positive_class_label}' is encoded as: {sargassum_encoded_value}")
        except ValueError:
            print(f"Error: Positive class '{current_positive_class_label}' not found in labels: {class_names_str}.")
            return

        os.makedirs(output_dir_models, exist_ok=True)
        le_path = os.path.join(output_dir_models, 'label_encoder_classification.joblib')
        joblib.dump(le, le_path)
        print(f"LabelEncoder saved to: {le_path}")

    except Exception as e:
         print(f"Error encoding labels: {e}."); return

    print(f"Splitting data (Test size: {TEST_SIZE}, Seed: {SEED}, Stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    print("Scaling features using StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(output_dir_models, 'scaler_classification.joblib')
    try: joblib.dump(scaler, scaler_path); print(f"Scaler saved to: {scaler_path}")
    except Exception as e: print(f"Error saving scaler: {e}")

    # --- Representative dataset generators ---
    def representative_dataset_gen_cnn():
        for i in range(min(1000, len(X_train_scaled))):
            sample = X_train_scaled[i:i+1]
            cnn_sample = sample.reshape((sample.shape[0], sample.shape[1], 1))
            yield [cnn_sample.astype(np.float32)]

    def representative_dataset_gen_mlp():
        for i in range(min(1000, len(X_train_scaled))):
            sample = X_train_scaled[i:i+1]
            yield [sample.astype(np.float32)]


    results = {}
    overall_start_time = time.time()

    # --- Train Keras MLP ---
    print("\n--- Training and Evaluating Keras MLP Classifier (Mixed Precision) ---")
    mlp_train_start_time = time.time()
    mlp_classifier = None
    try:
        input_shape_mlp = (X_train_scaled.shape[1],)
        mlp_classifier = build_mlp_classifier(input_shape_mlp)
        mlp_classifier.summary()

        early_stopping_mlp = EarlyStopping(monitor='val_loss', patience=MLP_EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)

        history_mlp = mlp_classifier.fit(
            X_train_scaled, y_train,
            epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE,
            validation_split=MLP_VALIDATION_SPLIT,
            callbacks=[early_stopping_mlp], verbose=2
        )
        mlp_train_duration = time.time() - mlp_train_start_time

        y_pred_proba_mlp = mlp_classifier.predict(X_test_scaled).flatten()
        y_pred_labels_mlp = (y_pred_proba_mlp > 0.5).astype(int)

        acc_mlp = accuracy_score(y_test, y_pred_labels_mlp)
        prec_mlp = precision_score(y_test, y_pred_labels_mlp, average='weighted', zero_division=0)
        rec_mlp = recall_score(y_test, y_pred_labels_mlp, average='weighted', zero_division=0)
        f1_mlp = f1_score(y_test, y_pred_labels_mlp, average='weighted', zero_division=0)

        results['Keras_MLP_Mixed'] = {'Accuracy': acc_mlp, 'Precision (w)': prec_mlp, 'Recall (w)': rec_mlp, 'F1-Score (w)': f1_mlp, 'Train Time (s)': mlp_train_duration}
        print("Test Set Classification Metrics for Keras MLP (Mixed):")
        print(results['Keras_MLP_Mixed'])

        report_str_mlp = classification_report(y_test, y_pred_labels_mlp, target_names=class_names_str, zero_division=0, digits=4)
        print(f"\nClassification Report for Keras MLP (Mixed):\n{report_str_mlp}")
        plot_confusion_matrix(y_test, y_pred_labels_mlp, class_names_str, 'Keras_MLP_Mixed', output_dir_plots)

        model_path_mlp_mixed = os.path.join(output_dir_models, 'mlp_classifier_mixed.keras')
        mlp_classifier.save(model_path_mlp_mixed)
        print(f"Trained Keras MLP (Mixed) saved to: {model_path_mlp_mixed}")

        # --- Quantize MLP Model to INT8 and FLOAT16 ---
        fp32_mlp = get_fp32_model_with_weights(model_path_mlp_mixed)
        if fp32_mlp:
            model_path_mlp_int8 = os.path.join(output_dir_models, 'mlp_classifier_int8.tflite')
            quantize_model_int8(fp32_mlp, model_path_mlp_int8, representative_dataset_gen_mlp)

            model_path_mlp_f16 = os.path.join(output_dir_models, 'mlp_classifier_f16.tflite')
            quantize_model_f16(fp32_mlp, model_path_mlp_f16)

    except Exception as e:
        print(f"ERROR during Keras MLP training/evaluation: {e}")
        traceback.print_exc()
        results['Keras_MLP_Mixed'] = {'Accuracy': np.nan, 'Precision (w)': np.nan, 'Recall (w)': np.nan, 'F1-Score (w)': np.nan, 'Train Time (s)': np.nan}


    # --- Train 1D CNN ---
    print("\n--- Training and Evaluating 1D CNN Classifier (Mixed Precision) ---")
    cnn_train_start_time = time.time()
    cnn_classifier = None
    try:
        X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        input_shape_cnn = (X_train_cnn.shape[1], X_train_cnn.shape[2])

        cnn_classifier = build_cnn_classifier(input_shape_cnn)
        cnn_classifier.summary()

        early_stopping_cnn = EarlyStopping(monitor='val_loss', patience=CNN_EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)

        history_cnn = cnn_classifier.fit(
            X_train_cnn, y_train,
            epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
            validation_split=CNN_VALIDATION_SPLIT,
            callbacks=[early_stopping_cnn], verbose=2
        )
        cnn_train_duration = time.time() - cnn_train_start_time

        y_pred_proba_cnn = cnn_classifier.predict(X_test_cnn).flatten()
        y_pred_labels_cnn = (y_pred_proba_cnn > 0.5).astype(int)

        acc_cnn = accuracy_score(y_test, y_pred_labels_cnn)
        prec_cnn = precision_score(y_test, y_pred_labels_cnn, average='weighted', zero_division=0)
        rec_cnn = recall_score(y_test, y_pred_labels_cnn, average='weighted', zero_division=0)
        f1_cnn = f1_score(y_test, y_pred_labels_cnn, average='weighted', zero_division=0)

        results['1D_CNN_Mixed'] = {'Accuracy': acc_cnn, 'Precision (w)': prec_cnn, 'Recall (w)': rec_cnn, 'F1-Score (w)': f1_cnn, 'Train Time (s)': cnn_train_duration}
        print("Test Set Classification Metrics for 1D CNN (Mixed):")
        print(results['1D_CNN_Mixed'])

        report_str_cnn = classification_report(y_test, y_pred_labels_cnn, target_names=class_names_str, zero_division=0, digits=4)
        print(f"\nClassification Report for 1D CNN (Mixed):\n{report_str_cnn}")
        plot_confusion_matrix(y_test, y_pred_labels_cnn, class_names_str, '1D_CNN_Mixed', output_dir_plots)

        model_path_cnn_mixed = os.path.join(output_dir_models, 'cnn_classifier_mixed.keras')
        cnn_classifier.save(model_path_cnn_mixed)
        print(f"Trained CNN classifier (Mixed) saved to: {model_path_cnn_mixed}")

        # --- Quantize CNN Model to INT8 and FLOAT16 ---
        fp32_cnn = get_fp32_model_with_weights(model_path_cnn_mixed)
        if fp32_cnn:
            model_path_cnn_int8 = os.path.join(output_dir_models, 'cnn_classifier_int8.tflite')
            quantize_model_int8(fp32_cnn, model_path_cnn_int8, representative_dataset_gen_cnn)

            model_path_cnn_f16 = os.path.join(output_dir_models, 'cnn_classifier_f16.tflite')
            quantize_model_f16(fp32_cnn, model_path_cnn_f16)

    except Exception as e:
        print(f"ERROR during CNN training/evaluation: {e}")
        traceback.print_exc()
        results['1D_CNN_Mixed'] = {'Accuracy': np.nan, 'Precision (w)': np.nan, 'Recall (w)': np.nan, 'F1-Score (w)': np.nan, 'Train Time (s)': np.nan}


    overall_duration = time.time() - overall_start_time
    print(f"\nTotal training and evaluation time for all models: {overall_duration:.2f} seconds")

    results_df = pd.DataFrame(results).T
    cols_order = ['Accuracy', 'Precision (w)', 'Recall (w)', 'F1-Score (w)', 'Train Time (s)']
    final_cols_order = [col for col in cols_order if col in results_df.columns]

    print("\n--- Overall Test Set Performance (Classification Metrics) ---")
    if not results_df.empty:
        results_df = results_df[final_cols_order]
        try: print(results_df.to_string(float_format="%.4f"))
        except Exception: print(results_df)

        os.makedirs(output_dir_tables, exist_ok=True)
        results_table_path = os.path.join(output_dir_tables, 'classification_performance_summary.csv')
        try:
            results_df.to_csv(results_table_path, float_format='%.4f')
            print(f"Classification performance summary saved to: {results_table_path}")
        except Exception as e: print(f"Error saving performance table: {e}")
    else:
        print("No results to display or save.")

    print("\n--- CLASSIFICATION Model Training & Evaluation Complete ---")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Hardcoded Arguments ---
    SCRIPT_CSV_PATH = "sargassum_data.csv"
    SCRIPT_OUTPUT_DIR_MODELS = "output/models_classification"
    SCRIPT_OUTPUT_DIR_PLOTS = "output/plots_classification"
    SCRIPT_OUTPUT_DIR_TABLES = "output/tables_classification"
    SCRIPT_POSITIVE_CLASS_LABEL = 'sargassum' # Default positive class label

    # --- Ensure output directories exist ---
    os.makedirs(SCRIPT_OUTPUT_DIR_MODELS, exist_ok=True)
    os.makedirs(SCRIPT_OUTPUT_DIR_PLOTS, exist_ok=True)
    os.makedirs(SCRIPT_OUTPUT_DIR_TABLES, exist_ok=True)

    # --- Call the main function with hardcoded arguments ---
    train_evaluate_classifiers(SCRIPT_CSV_PATH,
                               SCRIPT_OUTPUT_DIR_MODELS,
                               SCRIPT_OUTPUT_DIR_PLOTS,
                               SCRIPT_OUTPUT_DIR_TABLES,
                               SCRIPT_POSITIVE_CLASS_LABEL)