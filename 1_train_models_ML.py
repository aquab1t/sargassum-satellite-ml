# 1_train_models_XGBoost_ONLY.py

# --- Set Matplotlib Backend (MUST BE BEFORE importing pyplot or seaborn) ---
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for non-interactive plotting to file

# --- Standard Imports ---
import pandas as pd
import numpy as np
import os
import joblib # For saving sklearn models/scaler
import time # For timing operations
import matplotlib.pyplot as plt # Import AFTER setting backend
import seaborn as sns
import traceback  # Import traceback for error logging

# --- Scikit-learn Imports (for Utilities and CV) ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.model_selection import StratifiedKFold as SklearnStratifiedKFold
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("XGBoost found. Model will be trained on GPU.")
except ImportError:
    XGB_AVAILABLE = False
    print("CRITICAL: XGBoost not found. This script requires XGBoost.")

# Configuration for Reproducibility
SEED = 42
np.random.seed(SEED) # For NumPy operations

# --- Configuration ---
TEST_SIZE = 0.3
N_SPLITS_CV = 5 # Number of folds for cross-validation
CLASS_COLUMN = 'class'

# --- Model Definitions & Hyperparameter Grids (CLASSIFICATION) ---
def get_classifier_models_and_params(y_train_for_xgb_scale_pos_weight=None, positive_label_encoded=1):
    """
    Returns dictionary with only the GPU-accelerated XGBoost model.
    """
    if not XGB_AVAILABLE:
        return {}, {}

    xgb_scale_pos_weight_val = 1.0 # Default if not calculated
    if y_train_for_xgb_scale_pos_weight is not None:
        count_neg = np.sum(y_train_for_xgb_scale_pos_weight != positive_label_encoded)
        count_pos = np.sum(y_train_for_xgb_scale_pos_weight == positive_label_encoded)
        if count_pos > 0:
            xgb_scale_pos_weight_val = count_neg / count_pos
        print(f"Calculated XGBoost scale_pos_weight: {xgb_scale_pos_weight_val:.2f} (for positive_label_encoded={positive_label_encoded})")

    models = {
        'XGBoost_GPU': xgb.XGBClassifier(random_state=SEED,
                                       device='cuda', # Use GPU for training
                                       tree_method='hist',
                                       eval_metric='logloss'),
    }

    params = {
        'XGBoost_GPU': {
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0],
            'scale_pos_weight': [xgb_scale_pos_weight_val]
        },
    }
    return models, params

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


# --- Main Training & Evaluation Function (CLASSIFICATION) ---
def train_evaluate_classifiers(csv_path, output_dir_models, output_dir_plots, output_dir_tables, current_positive_class_label):
    """Loads data, trains GPU-accelerated XGBoost model, evaluates, and saves."""

    if not XGB_AVAILABLE:
        print("CRITICAL: XGBoost not found. Aborting training script.")
        return

    print(f"--- Starting GPU XGBoost CLASSIFICATION Model Training & Evaluation ---")
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
    sargassum_encoded_value = -1 # Initialize
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

    # Data is now all NumPy arrays. XGBoost will handle moving to GPU.

    models, params = get_classifier_models_and_params(y_train, sargassum_encoded_value)
    trained_models = {}
    results = {}
    overall_start_time = time.time()

    # Loop will only run for XGBoost
    for name, model_template in models.items():
        print(f"\n--- Training and Tuning {name} Classifier (on GPU) ---")
        model_train_start_time = time.time()
        best_model = None
        grid_search_results = None

        if name not in params or not params[name]:
             print(f"Warning: No hyperparameters defined for {name}. Using defaults.")
             best_model = model_template
             try:
                 best_model.fit(X_train_scaled, y_train) # Fit on NumPy
             except Exception as e:
                 print(f"Error fitting {name} with defaults: {e}");
                 traceback.print_exc()
                 continue
        else:
            print("   -> Using sklearn GridSearchCV, sklearn StratifiedKFold, and NumPy data")
            cv_strategy = SklearnStratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=SEED)
            grid_search = SklearnGridSearchCV(
                model_template, params[name], cv=cv_strategy, scoring='f1_weighted', n_jobs=1, verbose=1
            )
            # Use CPU (NumPy) data for sklearn's CV splitter
            X_fit, y_fit = X_train_scaled, y_train
            X_eval = X_test_scaled # Evaluate on NumPy data

            try:
                grid_search.fit(X_fit, y_fit) # This will now work
                best_model = grid_search.best_estimator_
                grid_search_results = grid_search
                print(f"Best params for {name}: {grid_search.best_params_}")
                print(f"Best F1 (weighted, CV) for {name}: {grid_search.best_score_:.4f}")
            except Exception as e:
                print(f"Error during GridSearchCV for {name}: {e}");
                traceback.print_exc() # This will now work
                continue

        if best_model is None:
            print(f"Skipping evaluation for {name} as model training failed.")
            results[name] = {'Accuracy': np.nan, 'Precision (w)': np.nan, 'Recall (w)': np.nan, 'F1-Score (w)': np.nan, 'Train Time (s)': np.nan}
            continue

        trained_models[name] = best_model
        model_train_duration = time.time() - model_train_start_time

        try:
            # Predict on NumPy test data
            y_pred_labels = best_model.predict(X_eval)

            acc = accuracy_score(y_test, y_pred_labels)
            prec = precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_labels, average='weighted', zero_division=0)
            results[name] = {'Accuracy': acc, 'Precision (w)': prec, 'Recall (w)': rec, 'F1-Score (w)': f1, 'Train Time (s)': model_train_duration}
            print(f"Test Set Classification Metrics for {name}:")
            print(results[name])
            print(f"Training time for {name}: {model_train_duration:.2f} seconds")

            print(f"\nClassification Report for {name}:")
            report_str = classification_report(y_test, y_pred_labels, target_names=class_names_str, zero_division=0, digits=4) # type: ignore
            print(report_str)
            try:
                report_path = os.path.join(output_dir_tables, f"classification_report_{name.lower().replace(' ', '_').replace('.', '')}.txt")
                with open(report_path, 'w') as f:
                    f.write(f"Classification Report for {name} (Positive Class: '{current_positive_class_label}' encoded as {sargassum_encoded_value})\n")
                    if grid_search_results:
                        f.write(f"Best params: {grid_search_results.best_params_}\n") # type: ignore
                        f.write(f"Best CV F1 (weighted): {grid_search_results.best_score_:.4f}\n\n") # type: ignore
                    else:
                        f.write("Trained with default parameters.\n\n")
                    f.write(report_str)
                print(f"Classification report for {name} saved to {report_path}")
            except Exception as e: print(f"Error saving classification report for {name}: {e}")

            plot_confusion_matrix(y_test, y_pred_labels, class_names_str, name, output_dir_plots) # type: ignore
        except Exception as e:
            print(f"Error during evaluation/plotting for {name}: {e}")
            traceback.print_exc() # Use traceback here
            results[name] = {'Accuracy': np.nan, 'Precision (w)': np.nan, 'Recall (w)': np.nan, 'F1-Score (w)': np.nan, 'Train Time (s)': model_train_duration}

        model_path = os.path.join(output_dir_models, f'{name.lower().replace(" ", "_").replace(".", "")}_classifier.joblib')
        try: joblib.dump(best_model, model_path); print(f"Trained {name} classifier saved: {model_path}")
        except Exception as e: print(f"Error saving {name} model: {e}")

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
        results_table_path = os.path.join(output_dir_tables, 'classification_performance_summary_gpu.csv')
        try:
            results_df.to_csv(results_table_path, float_format='%.4f')
            print(f"Classification performance summary saved to: {results_table_path}")
        except Exception as e: print(f"Error saving performance table: {e}")
    else:
        print("No results to display or save.")

    print("\n--- CLASSIFICATION Model Training & Evaluation Complete ---")

# --- Main Execution ---
if __name__ == "__main__":
    SCRIPT_CSV_PATH = "sargassum_data.csv"
    SCRIPT_OUTPUT_DIR_MODELS = "output/models_classification"
    SCRIPT_OUTPUT_DIR_PLOTS = "output/plots_classification"
    SCRIPT_OUTPUT_DIR_TABLES = "output/tables_classification"
    SCRIPT_POSITIVE_CLASS_LABEL = 'sargassum'

    os.makedirs(SCRIPT_OUTPUT_DIR_MODELS, exist_ok=True)
    os.makedirs(SCRIPT_OUTPUT_DIR_PLOTS, exist_ok=True)
    os.makedirs(SCRIPT_OUTPUT_DIR_TABLES, exist_ok=True)

    train_evaluate_classifiers(SCRIPT_CSV_PATH,
                               SCRIPT_OUTPUT_DIR_MODELS,
                               SCRIPT_OUTPUT_DIR_PLOTS,
                               SCRIPT_OUTPUT_DIR_TABLES,
                               SCRIPT_POSITIVE_CLASS_LABEL)