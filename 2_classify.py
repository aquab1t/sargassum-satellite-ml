import rasterio
import rasterio.windows
import numpy as np
import joblib
import os
import glob
from tqdm import tqdm
import traceback
import math # Added for ceiling function
import time # Added for timing

# Set TensorFlow logging level before TensorFlow import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Dependency Imports & Availability Checks ---
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass # OpenCV availability checked where needed

TF_AVAILABLE = False
try:
    import tensorflow as tf
    # --- SET GPU MEMORY GROWTH ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Set memory growth for: {gpus}")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    TF_AVAILABLE = True
except ImportError:
    pass # TensorFlow availability checked where needed


# --- Configuration ---
MODEL_ROOT_DIR = "output/"
DATA_ROOT_DIR = "satellite_data/"
OUTPUT_DIR = "output/fractional_cover_maps/"

# --- MODIFIED: List of models to run (Comparing int8 vs f16 TFLite) ---
MODELS_TO_RUN = [
    'mlp_classifier_int8.tflite',
    'mlp_classifier_f16.tflite',
    'cnn_classifier_int8.tflite',
    'cnn_classifier_f16.tflite'
]

POSITIVE_CLASS_LABEL_CONFIG = 'sargassum'
DISABLE_L8_HARMONIZATION = False

TRAINING_BAND_ORDER = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']

L8_TO_S2_HARMONIZATION_COEFFS = {
    'Blue':  {'slope': 0.9778, 'intercept': 0.0048},
    'Green': {'slope': 1.0379, 'intercept': -0.0009},
    'Red':   {'slope': 1.0431, 'intercept': -0.0011},
    'NIR':   {'slope': 0.9043, 'intercept': 0.0040},
    'SWIR1': {'slope': 0.9872, 'intercept': -0.0001}
}

SENSOR_CONFIG = {
    'Landsat-8': {
        'name': 'Landsat-8',
        'bands_needed': {'B2': 30, 'B3': 30, 'B4': 30, 'B5': 30, 'B6': 30},
        'target_resolution': 30,
        'band_map_to_train': {'B2': 'Blue', 'B3': 'Green', 'B4': 'Red', 'B5': 'NIR', 'B6': 'SWIR1'},
        'scale': 0.0000275, 'offset': -0.2,
        'file_patterns': {'B2': ['*_B2.TIF'], 'B3': ['*_B3.TIF'], 'B4': ['*_B4.TIF'], 'B5': ['*_B5.TIF'], 'B6': ['*_B6.TIF']},
        'harmonization_coeffs': L8_TO_S2_HARMONIZATION_COEFFS
    },
    'Sentinel-2': {
        'name': 'Sentinel-2',
        'bands_needed': {
            'B02': 10, 'B03': 10, 'B04': 10,
            'B08': 10,
            'B8A': 20,
            'B11': 20
        },
        'target_resolution': 10,
        'band_map_to_train': {'B02': 'Blue', 'B03': 'Green', 'B04': 'Red', 'B08': 'NIR', 'B8A': 'NIR', 'B11': 'SWIR1'},
        'scale': 0.0001, 'offset': 0.0,
        'file_patterns': {
            'B02': ['**/R10m/*_B02_10m.jp2'],
            'B03': ['**/R10m/*_B03_10m.jp2'],
            'B04': ['**/R10m/*_B04_10m.jp2'],
            'B08': ['**/R10m/*_B08_10m.jp2'],
            'B8A': ['**/R20m/*_B8A_20m.jp2'],
            'B11': ['**/R20m/*_B11_20m.jp2']
        }
    }
}
NODATA_VALUE = -9999.0
OUTPUT_DTYPE = np.float32

# --- Helper Functions ---

def load_scaler_and_encoder(scaler_path, label_encoder_path, positive_class_label):
    scaler, le, positive_class_index = None, None, -1
    try:
        if not os.path.exists(scaler_path): raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from: {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None, None, -1

    try:
        if not os.path.exists(label_encoder_path): raise FileNotFoundError(f"LabelEncoder not found: {label_encoder_path}")
        le = joblib.load(label_encoder_path)
        print(f"LabelEncoder loaded from: {label_encoder_path}")
        positive_class_index = le.transform([str(positive_class_label)])[0]
        print(f"Positive class '{positive_class_label}' corresponds to index {positive_class_index}.")
    except ValueError:
        print(f"Error: Positive class '{positive_class_label}' not found in LabelEncoder classes: {le.classes_ if le else 'N/A'}.")
        return scaler, None, -1
    except Exception as e:
        print(f"Error loading LabelEncoder: {e}")
        return scaler, None, -1
    return scaler, le, positive_class_index

def find_scene_band_files(scene_path, scene_id, sensor_config):
    band_files = {}
    for band_id_sensor, pattern_list in sensor_config['file_patterns'].items():
        file_found = False
        for pattern_suffix in pattern_list:
            search_pattern = os.path.join(scene_path, pattern_suffix)
            use_recursive = '**' in pattern_suffix
            found_files = sorted(glob.glob(search_pattern, recursive=use_recursive))

            if found_files:
                band_files[band_id_sensor] = found_files[0]
                file_found = True
                break

    found_concepts = {sensor_config['band_map_to_train'][b] for b in band_files if b in sensor_config['band_map_to_train']}
    required_concepts = {'Blue', 'Green', 'Red', 'SWIR1'}
    has_nir = 'NIR' in found_concepts

    if not required_concepts.issubset(found_concepts) or not has_nir:
        missing_concepts = (required_concepts - found_concepts)
        if not has_nir: missing_concepts.add('NIR')
        print(f"Error: Cannot process scene {scene_id}. Missing files for core concepts: {missing_concepts}")
        return None

    return band_files

def classify_scene_generic_model(scene_id, band_files_dict, sensor_config_entry,
                                 model_obj, model_type, model_name_prefix,
                                 scaler, positive_class_index, output_dir_scene):

    sensor_name = sensor_config_entry['name']
    target_res = sensor_config_entry['target_resolution']

    output_filename = f"fractional_cover_{sensor_name}_{scene_id}_{target_res}m_{model_name_prefix}.tif"
    output_path = os.path.join(output_dir_scene, output_filename)

    if os.path.exists(output_path):
        print(f"    Output already exists: {output_path}. Skipping.")
        return output_path

    ref_band_id_candidates = [bid for bid, res_val in sensor_config_entry['bands_needed'].items() if res_val == target_res and bid in band_files_dict]
    if not ref_band_id_candidates:
        ref_band_id_candidates = list(b for b in sensor_config_entry['bands_needed'].keys() if b in band_files_dict)
    if not ref_band_id_candidates:
        print(f"    Error: No usable bands found for {scene_id}."); return None
    ref_band_id = ref_band_id_candidates[0]

    scene_specific_band_map = {}
    has_nir = False
    for sensor_band, concept in sensor_config_entry['band_map_to_train'].items():
        if sensor_band in band_files_dict:
            if concept == 'NIR':
                if not has_nir:
                    scene_specific_band_map[sensor_band] = concept
                    has_nir = True
            else:
                scene_specific_band_map[sensor_band] = concept

    base_crs, base_transform = None, None
    profile, width, height = None, None, None
    try:
        with rasterio.open(band_files_dict[ref_band_id]) as src_ref:
            profile = src_ref.profile.copy()
            profile.pop('blockxsize', None); profile.pop('blockysize', None); profile.pop('tiled', None)
            profile.update(dtype=OUTPUT_DTYPE, count=1, nodata=NODATA_VALUE, compress='lzw', driver='GTiff')
            print(f"\n  Processing Scene: {scene_id} with Model: {model_name_prefix} (Type: {model_type})")
            width, height = src_ref.width, src_ref.height
            base_crs, base_transform = src_ref.crs, src_ref.transform
            if not width or not height: raise ValueError(f"Invalid ref raster dims: {width}x{height}")
            print(f"    Reference Grid ({ref_band_id}): {width}x{height}px (TargetRes={target_res}m)")
    except Exception as e:
        print(f"    Error reading profile from ref band {band_files_dict[ref_band_id]} for {scene_id}: {e}"); traceback.print_exc(); return None

    if base_crs is None or base_transform is None:
        print(f"    Error: Could not determine base CRS or transform for {scene_id}."); return None

    print(f"    Loading all scene bands into memory to speed up processing...")
    full_bands_data = {}
    full_bands_nodata = {}
    bands_to_load = list(scene_specific_band_map.keys())

    try:
        for sensor_band_id in bands_to_load:
            band_file_path = band_files_dict.get(sensor_band_id)
            if not band_file_path:
                print(f"    Error: Path for {sensor_band_id} missing."); return None
            with rasterio.open(band_file_path) as band_src:
                full_bands_data[sensor_band_id] = band_src.read(1)
                nodata_val = band_src.nodata
                if nodata_val is None: nodata_val = 0
                full_bands_nodata[sensor_band_id] = int(nodata_val)
        print("    All bands loaded.")
    except Exception as e:
        print(f"    Error loading full band data into RAM: {e}. Check system memory."); traceback.print_exc(); return None


    try:
        # --- Create a single performance timer ---
        total_prediction_time = 0
        total_pixels_processed = 0

        with rasterio.open(output_path, 'w', **profile) as dst:
            with rasterio.open(band_files_dict[ref_band_id]) as src_ref_for_windows:
                 windows_to_process = list(src_ref_for_windows.block_windows(1))

            for _, window in tqdm(windows_to_process, desc=f"  Classifying {scene_id} ({model_name_prefix})", unit="block", leave=False, mininterval=1.0):
                if window.width == 0 or window.height == 0: continue

                win_height, win_width = window.height, window.width
                output_block_final = np.full((win_height, win_width), NODATA_VALUE, dtype=OUTPUT_DTYPE)

                window_bands_raw = {}
                current_window_valid_mask = None
                all_bands_read_ok = True

                for sensor_band_id in bands_to_load:
                    try:
                        band_res = sensor_config_entry['bands_needed'][sensor_band_id]
                        ratio = band_res / target_res

                        if ratio == 1.0:
                            window_slice = (slice(window.row_off, window.row_off + win_height),
                                            slice(window.col_off, window.col_off + win_width))
                        else:
                            src_row_off = int(window.row_off // ratio)
                            src_col_off = int(window.col_off // ratio)
                            src_row_end = int(math.ceil((window.row_off + win_height) / ratio))
                            src_col_end = int(math.ceil((window.col_off + win_width) / ratio))

                            window_slice = (slice(src_row_off, src_row_end),
                                            slice(src_col_off, src_col_end))

                        data_native_res = full_bands_data[sensor_band_id][window_slice]

                        if ratio != 1.0:
                            if not CV2_AVAILABLE:
                                print(f"\n    Error: OpenCV needed for resampling {sensor_band_id}.");
                                all_bands_read_ok=False; break
                            if data_native_res.size == 0:
                                all_bands_read_ok=False; break # Skip this block

                            data_for_resize = data_native_res.astype(np.float32, copy=False)
                            data = cv2.resize(data_for_resize, (win_width, win_height), interpolation=cv2.INTER_LANCZOS4)
                        else:
                            data = data_native_res

                        window_bands_raw[sensor_band_id] = data
                        nodata_val_band = full_bands_nodata[sensor_band_id]
                        band_mask = (data.astype(np.int64) != nodata_val_band)
                        current_window_valid_mask = band_mask if current_window_valid_mask is None else (current_window_valid_mask & band_mask)

                    except Exception as e:
                        print(f"\n    Error slicing/resizing {sensor_band_id} win {window}: {e}");
                        all_bands_read_ok=False;
                        break

                if not all_bands_read_ok or current_window_valid_mask is None or not current_window_valid_mask.any():
                    dst.write(output_block_final, indexes=1, window=window); continue

                n_valid_pixels_in_window = np.sum(current_window_valid_mask)
                scaled_pixel_spectra = {}
                sensor_scale, sensor_offset = sensor_config_entry['scale'], sensor_config_entry['offset']
                for s_bid, raw_data_arr in window_bands_raw.items():
                    valid_raw_data = raw_data_arr[current_window_valid_mask]
                    sr_values = (valid_raw_data.astype(np.float64) * sensor_scale) + sensor_offset
                    scaled_pixel_spectra[s_bid] = sr_values

                if sensor_name == 'Landsat-8' and 'harmonization_coeffs' in sensor_config_entry and not DISABLE_L8_HARMONIZATION:
                    harm_coeffs_sensor = sensor_config_entry['harmonization_coeffs']
                    for sensor_band_id_l8, common_train_name in scene_specific_band_map.items():
                        if common_train_name in harm_coeffs_sensor and sensor_band_id_l8 in scaled_pixel_spectra:
                            coeffs = harm_coeffs_sensor[common_train_name]
                            scaled_pixel_spectra[sensor_band_id_l8] = \
                                (coeffs['slope'] * scaled_pixel_spectra[sensor_band_id_l8]) + coeffs['intercept']

                model_input_features_valid_pixels = np.full((n_valid_pixels_in_window, len(TRAINING_BAND_ORDER)), np.nan, dtype=np.float32)
                for i_train_band, train_band_concept in enumerate(TRAINING_BAND_ORDER):
                    source_sensor_band_id = next((s_bid for s_bid, concept in scene_specific_band_map.items() if concept == train_band_concept), None)
                    if source_sensor_band_id and source_sensor_band_id in scaled_pixel_spectra:
                        model_input_features_valid_pixels[:, i_train_band] = scaled_pixel_spectra[source_sensor_band_id]

                final_feature_validity_mask = ~np.isnan(model_input_features_valid_pixels).any(axis=1)
                if not final_feature_validity_mask.any():
                    dst.write(output_block_final, indexes=1, window=window); continue

                model_input_final = model_input_features_valid_pixels[final_feature_validity_mask]
                try:
                    features_for_model_scaled = scaler.transform(model_input_final)
                except Exception as e: print(f"\n    Scaler error: {e}"); traceback.print_exc(); dst.write(output_block_final, indexes=1, window=window); continue

                predictions_proba_for_valid_features = np.full(features_for_model_scaled.shape[0], np.nan, dtype=OUTPUT_DTYPE)

                try:
                    input_data = features_for_model_scaled
                    is_cnn = 'cnn' in model_name_prefix.lower()
                    if is_cnn:
                        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))

                    # --- Start Timer ---
                    start_time = time.perf_counter()

                    if model_type == 'keras':
                        # This type is no longer loaded by default, but keeping logic
                        pred_p_all_classes = model_obj.predict(input_data.astype(np.float32), verbose=0)

                    elif model_type == 'tflite':
                        interpreter = model_obj
                        input_details = interpreter.get_input_details()[0]
                        output_details = interpreter.get_output_details()[0]

                        current_batch_size = input_data.shape[0]
                        if input_details['shape'][0] != current_batch_size:
                            interpreter.resize_tensor_input(input_details['index'], [current_batch_size] + list(input_details['shape'][1:]))
                            interpreter.allocate_tensors()

                        interpreter.set_tensor(input_details['index'], input_data.astype(input_details['dtype']))
                        interpreter.invoke()
                        pred_p_all_classes = interpreter.get_tensor(output_details['index'])

                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")

                    # --- End Timer ---
                    end_time = time.perf_counter()
                    total_prediction_time += (end_time - start_time)
                    total_pixels_processed += input_data.shape[0]

                    predictions_proba_for_valid_features = pred_p_all_classes.flatten()

                except Exception as e: print(f"\n    Prediction error ({model_name_prefix}): {e}"); traceback.print_exc(); dst.write(output_block_final, indexes=1, window=window); continue

                temp_preds_on_initial_valid_mask = np.full(n_valid_pixels_in_window, NODATA_VALUE, dtype=OUTPUT_DTYPE)
                temp_preds_on_initial_valid_mask[final_feature_validity_mask] = predictions_proba_for_valid_features
                output_block_final[current_window_valid_mask] = temp_preds_on_initial_valid_mask
                output_block_final[np.isnan(output_block_final) | np.isinf(output_block_final)] = NODATA_VALUE
                dst.write(output_block_final, indexes=1, window=window)

        # --- After loop, print performance ---
        if total_pixels_processed > 0 and total_prediction_time > 0:
            avg_time_us = (total_prediction_time / total_pixels_processed) * 1_000_000 # Time per pixel in microseconds
            pixels_per_sec = total_pixels_processed / total_prediction_time
            print(f"    Inference Performance ({model_name_prefix}):")
            print(f"      Total Pixels: {total_pixels_processed:,}")
            print(f"      Total Time: {total_prediction_time:.2f} s")
            print(f"      Pixels/sec: {pixels_per_sec:,.0f}")
            print(f"      Avg. Time/pixel: {avg_time_us:.2f} µs")

        print(f"  Fractional cover map ({model_name_prefix}, {target_res}m) saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"!!! Unexpected Error processing {scene_id} with {model_name_prefix}: {e}")
        traceback.print_exc()
        if os.path.exists(output_path):
             try: os.remove(output_path); print(f"    Removed incomplete file: {output_path}")
             except OSError as oe: print(f"    Error removing file {output_path}: {oe}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Classification (TFLite INT8 vs F16) ---")
    print(f"Models to run: {', '.join(MODELS_TO_RUN)}")
    print(f"Data Root: {DATA_ROOT_DIR}, Output: {OUTPUT_DIR}")
    print(f"Positive Class: {POSITIVE_CLASS_LABEL_CONFIG}, L8 Harmonization Disabled: {DISABLE_L8_HARMONIZATION}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not TF_AVAILABLE:
        print("CRITICAL: TensorFlow not available. Script cannot run.")
        exit(1)

    models_artifacts_subdir = "models_classification"
    scaler_path = os.path.join(MODEL_ROOT_DIR, models_artifacts_subdir, "scaler_classification.joblib")
    label_encoder_path = os.path.join(MODEL_ROOT_DIR, models_artifacts_subdir, "label_encoder_classification.joblib")

    scaler, le, positive_class_index = load_scaler_and_encoder(scaler_path, label_encoder_path, POSITIVE_CLASS_LABEL_CONFIG)
    if scaler is None or le is None or positive_class_index == -1:
        print("Critical error: Scaler or LabelEncoder failed to load, or positive class index invalid. Exiting.")
        exit(1)

    # --- MODIFIED: Model Loading Loop ---
    loaded_models_info = []
    for model_filename in MODELS_TO_RUN:
        model_full_path = os.path.join(MODEL_ROOT_DIR, models_artifacts_subdir, model_filename)
        model_name_prefix = os.path.splitext(model_filename)[0]

        if not os.path.exists(model_full_path):
            print(f"Warning: Model file not found, skipping: {model_full_path}")
            continue

        try:
            if model_filename.endswith('.tflite'):
                model_obj = tf.lite.Interpreter(model_path=model_full_path)
                model_obj.allocate_tensors() # Allocate once at load time
                model_type = 'tflite'
                print(f"Successfully loaded TFLite model: {model_filename}")
            else:
                print(f"Warning: Skipping non-TFLite model: {model_filename}")
                continue

            loaded_models_info.append({
                'name_prefix': model_name_prefix,
                'model_obj': model_obj,
                'type': model_type
            })
        except Exception as e:
            print(f"Error loading model {model_filename}: {e}")
            traceback.print_exc()

    if not loaded_models_info:
        print("Error: No TFLite models were successfully loaded. Exiting.")
        exit(1)

    print(f"\nSuccessfully loaded {len(loaded_models_info)} TFLite models. Starting classification...")
    # --- End Modified Model Loading Loop ---


    if DISABLE_L8_HARMONIZATION:
        if 'Landsat-8' in SENSOR_CONFIG and 'harmonization_coeffs' in SENSOR_CONFIG['Landsat-8']:
            del SENSOR_CONFIG['Landsat-8']['harmonization_coeffs']
            print("Landsat-8 to Sentinel-2 harmonization DISABLED.")
    else:
        if 'Landsat-8' in SENSOR_CONFIG:
            if 'harmonization_coeffs' not in SENSOR_CONFIG['Landsat-8']:
                 SENSOR_CONFIG['Landsat-8']['harmonization_coeffs'] = L8_TO_S2_HARMONIZATION_COEFFS
            print("Landsat-8 to Sentinel-2 harmonization ENABLED.")

    sensors_to_process = ['Landsat-8', 'Sentinel-2']

    for model_info in loaded_models_info:
        all_successful_outputs = []
        model_name = model_info['name_prefix']
        print(f"\n\n===== Processing ALL scenes with Model: {model_name} (Type: {model_info['type']}) =====")

        for sensor_key_config in sensors_to_process:
            if sensor_key_config not in SENSOR_CONFIG:
                print(f"Warning: Config for sensor '{sensor_key_config}' not found. Skipping.")
                continue
            sensor_config_current_iter = SENSOR_CONFIG[sensor_key_config]
            sensor_display_name = sensor_config_current_iter['name']

            print(f"\n===== Processing Sensor: {sensor_display_name} =====")
            if sensor_display_name == 'Landsat-8': sensor_data_subdir_name = 'landsat'
            elif sensor_display_name == 'Sentinel-2': sensor_data_subdir_name = 'sentinel'
            else: sensor_data_subdir_name = sensor_display_name.lower().replace('-', '')

            current_sensor_root_path = os.path.join(DATA_ROOT_DIR, sensor_data_subdir_name)
            if not os.path.isdir(current_sensor_root_path):
                print(f"Warning: Data directory for {sensor_display_name} not found: {current_sensor_root_path}. Skipping."); continue

            ignore_dirs = ["warped_10m", "warped_20m", "warped_30m", "warped_250m", "aligned_250m", "cloudmask", "temp", ".ipynb_checkpoints", "classification_results", "fractional_cover_maps"]
            potential_scene_dirs = [
                d for d in os.listdir(current_sensor_root_path)
                if os.path.isdir(os.path.join(current_sensor_root_path, d)) and
                   not d.startswith('.') and d.lower() not in ignore_dirs
            ]

            print(f"Found {len(potential_scene_dirs)} potential scenes/days for {sensor_display_name}.")
            if not potential_scene_dirs:
                print(f"No scenes/days to process for {sensor_display_name}."); continue

            for scene_id_name in sorted(potential_scene_dirs):
                current_scene_path = os.path.join(current_sensor_root_path, scene_id_name)
                band_files_for_scene = find_scene_band_files(current_scene_path, scene_id_name, sensor_config_current_iter)
                if not band_files_for_scene:
                    print(f"Skipping scene {scene_id_name} ({sensor_display_name}) due to missing band files."); continue

                output_file_path = classify_scene_generic_model(
                    scene_id_name, band_files_for_scene, sensor_config_current_iter,
                    model_info['model_obj'], model_info['type'], model_info['name_prefix'],
                    scaler, positive_class_index, OUTPUT_DIR
                )
                if output_file_path: all_successful_outputs.append(output_file_path)

        print(f"\n===== Classification Processing Complete for {model_name} =====")
        if all_successful_outputs:
            print("Successfully generated the following fractional cover maps:")
            for f_path in sorted(all_successful_outputs): print(f" - {f_path}")
        else: print("No output maps were generated successfully for this model.")

    print("\n\n===== ALL MODEL CLASSIFICATION RUNS COMPLETE =====")