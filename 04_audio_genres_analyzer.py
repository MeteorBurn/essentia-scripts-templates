#!/usr/-bin/env python3
# -*- coding: utf-8 -*-

import os

# Suppress TensorFlow INFO messages before importing essentia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import traceback
from typing import List, Dict, Optional, Tuple, Any, Set
import subprocess
import tempfile
import sys

# Import Essentia after setting the TF log level
try:
    import essentia
    import essentia.standard as es
except ImportError:
    print("ERROR: Essentia library not found.")
    print("Please install Essentia (e.g., pip install essentia)")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Could not import Essentia: {e}")
    sys.exit(1)


# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
CONFIG = {
    # --- Logging ---
    "loglevel": "INFO",  # Levels: DEBUG, INFO, WARNING, ERROR

    # --- Input/Output ---
    "input_directory": "/mnt/m/Abstracted",
    "output_directory": "abstracted_results",
    "file_pattern": "",
    "max_files": 0,

    # --- Conversion Parameters ---
    "convert_to_wav": False,
    "ffmpeg_path": "ffmpeg",
    "temp_dir": "",

    # --- Audio Processing Parameters ---
    "audio_offset": 60,
    "audio_duration": 30,

    # --- Essentia Parameters ---
    "sample_rate": 16000,
    "resample_quality": 4,
    "effnet_patch_hop_size": 128,

    # --- Model Paths ---
    "models_dir": "models",
    "embedding_model_filename": "discogs-effnet-bs64-1.pb",

    # --- Classifier Models ---
    # Define classifiers and their specific parameters, including config and model files.
    "classifier_models": {
        "genre_discogs400": {
            "enabled": True,
            "config_filename": "genre_discogs400-discogs-effnet-1.json",
            "model_filename": "genre_discogs400-discogs-effnet-1.pb",
            "num_genres": 3,
            "input_node": "serving_default_model_Placeholder",
            "output_node": "PartitionedCall:0"
        },
        "mtg_jamendo_moodtheme": {
            "enabled": True,
            "config_filename": "mtg_jamendo_moodtheme-discogs-effnet-1.json",
            "model_filename": "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
            "num_mood_tags": 3
        },
        "mtg_jamendo_instrument": {
            "enabled": True,
            "config_filename": "mtg_jamendo_instrument-discogs-effnet-1.json",
            "model_filename": "mtg_jamendo_instrument-discogs-effnet-1.pb",
            "num_instrument_tags": 3
        },
        "mtt_discogs_effnet": {
            "enabled": True,
            "config_filename": "mtt-discogs-effnet-1.json",
            "model_filename": "mtt-discogs-effnet-1.pb",
            "num_mtt_tags": 3
        },
        "danceability_discogs_effnet": {
            "enabled": False,
            "config_filename": "danceability-discogs-effnet-1.json",
            "model_filename": "danceability-discogs-effnet-1.pb",
            "output_node": "model/Softmax",
            "danceability_threshold": 0.5
        },
        "mood_aggressive_discogs_effnet": {
            "enabled": False,
            "config_filename": "mood_aggressive-discogs-effnet-1.json",
            "model_filename": "mood_aggressive-discogs-effnet-1.pb",
            "output_node": "model/Softmax",
            "mood_aggressive_threshold": 0.5
        },
        "mood_happy_discogs_effnet": {
            "enabled": False,
            "config_filename": "mood_happy-discogs-effnet-1.json",
            "model_filename": "mood_happy-discogs-effnet-1.pb",
            "output_node": "model/Softmax",
            "mood_happy_threshold": 0.5
        },
        "mood_party_discogs_effnet": {
            "enabled": False,
            "config_filename": "mood_party-discogs-effnet-1.json",
            "model_filename": "mood_party-discogs-effnet-1.pb",
            "output_node": "model/Softmax",
            "mood_party_threshold": 0.5
        },
        "mood_relaxed_discogs_effnet": {
            "enabled": False,
            "config_filename": "mood_relaxed-discogs-effnet-1.json",
            "model_filename": "mood_relaxed-discogs-effnet-1.pb",
            "output_node": "model/Softmax",
            "mood_relaxed_threshold": 0.5
        },
        "mood_sad_discogs_effnet": {
            "enabled": False,
            "config_filename": "mood_sad-discogs-effnet-1.json",
            "model_filename": "mood_sad-discogs-effnet-1.pb",
            "output_node": "model/Softmax",
            "mood_sad_threshold": 0.5
        },
        "voice_instrumental_discogs_effnet": {
            "enabled": False,
            "config_filename": "voice_instrumental-discogs-effnet-1.json",
            "model_filename": "voice_instrumental-discogs-effnet-1.pb",
            "output_node": "model/Softmax",
            "voice_instrumental_threshold": 0.5
        },
        "gender_discogs_effnet": {
            "enabled": False,
            "config_filename": "gender-discogs-effnet-1.json",
            "model_filename": "gender-discogs-effnet-1.pb",
            "output_node": "model/Softmax",
            "gender_threshold": 0.5
        }
    },

    # --- MAEST Models (end-to-end) ---
    "maest_models": {
        "maest_400_v2": {
            "enabled": True,
            "config_filename": "discogs-maest-30s-pw-2.json",
            "model_filename": "discogs-maest-30s-pw-2.pb",
            "input_node": "melspectrogram",
            "output_node": "PartitionedCall/Identity_13",
            "num_genres": 3
        }
    }
}

# =============================================================================
# CONSTANTS
# =============================================================================
MODELS_BASE_PATH = Path(CONFIG["models_dir"])
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif', '.m4a']
ClassificationResult = Dict[str, Any]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def setup_logging(level: str):
    """Initializes the global logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Logging level set to: {level.upper()}")

def find_audio_files(directory: str, file_pattern: str = "") -> List[Path]:
    """Finds all audio files in a directory recursively and sorts them."""
    audio_files = []
    start_dir = Path(directory)
    pattern_lower = file_pattern.lower() if file_pattern else None
    if not start_dir.is_dir():
        logging.error(f"Input directory not found: {start_dir}")
        return []
    logging.info(f"Searching for all audio files in {start_dir}...")
    for item in start_dir.rglob('*'):
        if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
            if pattern_lower and pattern_lower not in item.stem.lower():
                continue
            audio_files.append(item)
    
    audio_files.sort()
    logging.info(f"Found {len(audio_files)} total audio files.")
    return audio_files

def get_existing_json_stems(directory: str) -> Set[str]:
    """Returns a set of filenames for existing JSON results to avoid re-processing."""
    json_stems = set()
    output_dir = Path(directory)
    if not output_dir.is_dir():
        return json_stems
    for item in output_dir.glob('*.json'):
        if item.is_file():
            json_stems.add(item.stem)
    return json_stems

def load_json_config(json_path: Path) -> Optional[Dict]:
    """Loads a model's JSON configuration file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Could not load or parse JSON {json_path.name}: {e}")
        return None

def process_labels(raw_labels: List[str]) -> List[str]:
    """Cleans up label strings from model configs."""
    return [label.split('---', 1)[-1] for label in raw_labels]

def trim_audio_segment(audio: np.ndarray, sample_rate: int, offset_sec: float, duration_sec: float) -> np.ndarray:
    """Extracts a segment from an audio buffer."""
    if audio is None or len(audio) == 0: return audio
    start_sample = int(offset_sec * sample_rate)
    end_sample = start_sample + int(duration_sec * sample_rate)
    if start_sample >= len(audio):
        start_sample = 0
        end_sample = int(duration_sec * sample_rate)
    if end_sample > len(audio):
        end_sample = len(audio)
    return audio[start_sample:end_sample]

def convert_audio_to_wav(input_path: Path, target_sr: int, ffmpeg_exec: str, temp_dir: Optional[str]) -> Optional[Path]:
    """Converts any audio file to a temporary WAV file using ffmpeg."""
    temp_wav_path: Optional[Path] = None
    try:
        fd, temp_wav_path_str = tempfile.mkstemp(suffix=".wav", prefix="convert_", dir=temp_dir)
        os.close(fd)
        temp_wav_path = Path(temp_wav_path_str)
        command = [
            ffmpeg_exec, '-hide_banner', '-loglevel', 'error',
            '-i', str(input_path.resolve()), '-y',
            '-acodec', 'pcm_s16le', '-ar', str(target_sr), '-ac', '1',
            str(temp_wav_path.resolve())
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        if temp_wav_path.exists() and temp_wav_path.stat().st_size > 0:
            return temp_wav_path
        return None
    except Exception as e:
        logging.error(f"ffmpeg conversion error for {input_path.name}: {e}")
        if temp_wav_path and temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True)
        return None

def get_platform_paths(file_path: Path) -> Dict[str, str]:
    """Converts a file path to Windows and WSL formats if applicable."""
    wsl_path = file_path.as_posix()
    win_path = wsl_path
    parts = file_path.parts
    if len(parts) > 2 and parts[0] == '/' and parts[1] == 'mnt':
        drive_letter = parts[2]
        if len(drive_letter) == 1 and 'a' <= drive_letter.lower() <= 'z':
            path_body = "/".join(parts[3:])
            win_path = f"{drive_letter.upper()}:/{path_body}"
    return {"win": win_path, "wsl": wsl_path}

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models(config: Dict) -> Optional[Dict[str, Any]]:
    """
    Loads the base embedding model, classifier models, and MAEST models.
    """
    models: Dict[str, Any] = {'classifiers': {}, 'maest': {}}

    # 1. Load base embedding model for classifiers
    if config.get("embedding_model_filename"):
        logging.info("Loading base embedding model (EffNet)...")
        embedding_model_path = MODELS_BASE_PATH / config["embedding_model_filename"]
        if not embedding_model_path.is_file():
            logging.critical(f"Embedding model not found: {embedding_model_path}")
            return None
        try:
            models['embedding_model'] = es.TensorflowPredictEffnetDiscogs(
                graphFilename=str(embedding_model_path),
                output="PartitionedCall:1",
                patchHopSize=config["effnet_patch_hop_size"]
            )
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.critical(f"Failed to load embedding model: {e}")
            return None

    # 2. Load specified classifier models from config
    logging.info("Loading specified classifier models...")
    for model_name, model_params in config.get("classifier_models", {}).items():
        if not model_params.get("enabled", False):
            continue
        model_path = MODELS_BASE_PATH / model_params["model_filename"]
        if not model_path.is_file():
            logging.warning(f"Model file for '{model_name}' not found at {model_path}. Skipping.")
            continue
        json_path = MODELS_BASE_PATH / model_params["config_filename"]
        model_config = load_json_config(json_path)
        if not model_config:
            logging.warning(f"Config for '{model_name}' not found or failed to load. Skipping.")
            continue
        try:
            model_args = {"graphFilename": str(model_path)}
            if model_params.get("input_node"):
                model_args["input"] = model_params["input_node"]
            if model_params.get("output_node"):
                model_args["output"] = model_params["output_node"]
            classifier_model = es.TensorflowPredict2D(**model_args)
            labels = process_labels(model_config.get('classes', []))
            models['classifiers'][model_name] = {
                'model': classifier_model,
                'labels': labels,
            }
            logging.info(f" -> Loaded classifier: '{model_name}' with {len(labels)} labels.")
        except Exception as e:
            logging.error(f"Failed to load classifier '{model_name}': {e}")
            logging.debug(traceback.format_exc())

    # 3. Load MAEST models
    logging.info("Loading MAEST models...")
    for model_name, model_params in config.get("maest_models", {}).items():
        if not model_params.get("enabled", False):
            continue
        model_path = MODELS_BASE_PATH / model_params["model_filename"]
        if not model_path.is_file():
            logging.warning(f"Model file for '{model_name}' not found at {model_path}. Skipping.")
            continue
        json_path = MODELS_BASE_PATH / model_params["config_filename"]
        model_config = load_json_config(json_path)
        if not model_config:
            logging.warning(f"Config for '{model_name}' not found or failed to load. Skipping.")
            continue
        try:
            maest_model = es.TensorflowPredictMAEST(
                graphFilename=str(model_path),
                input=model_params["input_node"],
                output=model_params["output_node"]
            )
            labels = process_labels(model_config.get('classes', []))
            models['maest'][model_name] = {
                'model': maest_model,
                'labels': labels,
            }
            logging.info(f" -> Loaded MAEST model: '{model_name}' with {len(labels)} labels.")
        except Exception as e:
            logging.error(f"Failed to load MAEST model '{model_name}': {e}")
            logging.debug(traceback.format_exc())

    if not models['classifiers'] and not models['maest']:
        logging.warning("No active models were loaded. The script will only extract basic file info.")

    return models

# =============================================================================
# PREDICTION PROCESSING
# =============================================================================
def process_predictions(
    predictions: np.ndarray, labels: List[str], top_n: int
) -> List[Tuple[str, float]]:
    """Sorts predictions and returns the top N results."""
    if predictions is None:
        return []

    processed_preds = np.mean(predictions, axis=0) if predictions.ndim > 1 else predictions
    predictions_final = np.squeeze(processed_preds)

    if len(predictions_final) != len(labels):
        logging.warning(f"Prediction/label size mismatch ({len(predictions_final)} vs {len(labels)}).")
        return []

    probabilities = sorted(zip(labels, predictions_final), key=lambda item: item[1], reverse=True)
    return [(label, float(prob)) for label, prob in probabilities[:top_n]]

# =============================================================================
# CORE ANALYSIS FUNCTION
# =============================================================================
def analyze_audio_file(
    original_audio_path: Path,
    models: Dict[str, Any],
    config: Dict,
    output_dir: Path
) -> Optional[ClassificationResult]:
    """
    Analyzes a single audio file with both EffNet classifiers and MAEST models.
    """
    path_to_analyze = original_audio_path
    temp_wav_file: Optional[Path] = None
    needs_cleanup = False

    if config["convert_to_wav"] and original_audio_path.suffix.lower() != '.wav':
        temp_wav_file = convert_audio_to_wav(
            original_audio_path, config["sample_rate"], config["ffmpeg_path"], config.get("temp_dir")
        )
        if temp_wav_file:
            path_to_analyze, needs_cleanup = temp_wav_file, True
        else:
            return None

    json_data: ClassificationResult = {
        "file_path": get_platform_paths(original_audio_path),
        "file_name": original_audio_path.stem,
        "file_extension": original_audio_path.suffix,
        "timestamp": datetime.now().isoformat(),
        "analysis_config": {
            "audio_segment_offset": config["audio_offset"],
            "audio_segment_duration": config["audio_duration"],
        },
        "analysis_results": {},
    }

    try:
        loader = es.MonoLoader(filename=str(path_to_analyze), sampleRate=config["sample_rate"], resampleQuality=config["resample_quality"])
        audio = loader()
        if audio is None or len(audio) < config["sample_rate"] * 0.1:
            raise ValueError("Audio is empty or too short.")

        audio = trim_audio_segment(audio, config["sample_rate"], config["audio_offset"], config["audio_duration"])
        if audio is None or len(audio) < config["sample_rate"] * 0.1:
            raise ValueError("Audio segment is too short after trimming.")

        all_results = {}

        # --- Step 1: MAEST-based models (now first) ---
        if models.get('maest'):
            for name, maest_data in models['maest'].items():
                try:
                    model_params = config.get("maest_models", {}).get(name, {})
                    preds = maest_data['model'](audio) # MAEST uses audio directly
                    top_n = model_params.get("num_genres", 5)
                    structured_result = process_predictions(preds, maest_data['labels'], top_n)
                    all_results[name] = {
                        "labels": [label for label, score in structured_result],
                        "confidences": [round(score, 4) for label, score in structured_result]
                    }
                except Exception as maest_err:
                    logging.error(f"Error in MAEST model '{name}': {maest_err}")
                    all_results[name] = {"error": str(maest_err)}

        # --- Step 2: EffNet-based classifiers (now second) ---
        if models.get('embedding_model') and models.get('classifiers'):
            logging.debug(f"Calculating EffNet embeddings for {original_audio_path.name}...")
            full_embeddings = models['embedding_model'](audio)
            if full_embeddings is None:
                raise ValueError("Embedding model returned None.")

            # --- Step 2a: Process EffNet Genre classifier first ---
            genre_model_name = "genre_discogs400"
            if genre_model_name in models['classifiers']:
                try:
                    classifier_data = models['classifiers'][genre_model_name]
                    model_params = config.get("classifier_models", {}).get(genre_model_name, {})
                    preds = classifier_data['model'](full_embeddings)
                    top_n_key = next((key for key in model_params if key.startswith('num_')), None)
                    top_n = model_params.get(top_n_key, 3)
                    structured_result = process_predictions(preds, classifier_data['labels'], top_n)
                    all_results[genre_model_name] = {
                        "labels": [label for label, score in structured_result],
                        "confidences": [round(score, 4) for label, score in structured_result]
                    }
                except Exception as classifier_err:
                    logging.error(f"Error in classifier '{genre_model_name}': {classifier_err}")
                    all_results[genre_model_name] = {"error": str(classifier_err)}
            
            # --- Step 2b: Process other classifiers ---
            for name, classifier_data in models['classifiers'].items():
                if name == genre_model_name:
                    continue # Already processed
                try:
                    model_params = config.get("classifier_models", {}).get(name, {})
                    preds = classifier_data['model'](full_embeddings)
                    threshold_key = next((key for key in model_params if key.endswith('_threshold')), None)
                    if threshold_key:
                        labels = classifier_data['labels']
                        positive_class = [l for l in labels if not l.startswith('not_') and not l.startswith('non_')][0]
                        positive_idx = labels.index(positive_class)
                        probability = np.mean(preds, axis=0)[positive_idx]
                        threshold = model_params.get(threshold_key, 0.5)
                        result_label = positive_class if probability >= threshold else f"not_{positive_class}"
                        all_results[name] = {
                            "result": result_label,
                            "confidence": round(float(probability), 4)
                        }
                    else:
                        top_n_key = next((key for key in model_params if key.startswith('num_')), None)
                        top_n = model_params.get(top_n_key, 3)
                        structured_result = process_predictions(preds, classifier_data['labels'], top_n)
                        all_results[name] = {
                            "labels": [label for label, score in structured_result],
                            "confidences": [round(score, 4) for label, score in structured_result]
                        }
                except Exception as classifier_err:
                    logging.error(f"Error in classifier '{name}': {classifier_err}")
                    all_results[name] = {"error": str(classifier_err)}

        json_data["analysis_results"] = all_results

    except Exception as e:
        logging.error(f"Failed to analyze {original_audio_path.name}: {e}")
        logging.debug(traceback.format_exc())
        json_data["error"] = str(e)

    finally:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            json_filename = output_dir / (original_audio_path.stem + ".json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        except Exception as json_err:
            logging.error(f"Could not save JSON for {original_audio_path.name}: {json_err}")

        if needs_cleanup and temp_wav_file and temp_wav_file.exists():
            temp_wav_file.unlink(missing_ok=True)

    return json_data

# =============================================================================
# NUMPY JSON ENCODER
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def format_log_summary(results: Dict) -> str:
    """Creates a simple log summary string from the analysis results."""
    summary_parts = []
    
    model_log_order = [
        "genre_discogs400", "maest_400_v2", "mtg_jamendo_moodtheme", "mtg_jamendo_instrument", "mtt_discogs_effnet", 
        "danceability_discogs_effnet", "mood_happy_discogs_effnet",
        "mood_sad_discogs_effnet", "mood_aggressive_discogs_effnet", "mood_relaxed_discogs_effnet",
        "mood_party_discogs_effnet", "voice_instrumental_discogs_effnet", "gender_discogs_effnet"
    ]

    log_labels = {
        "genre_discogs400": "Genre(EffNet)", "maest_400_v2": "Genre(MAEST)",
        "mtg_jamendo_instrument": "Instrument", "mtt_discogs_effnet": "Tag", 
        "mtg_jamendo_moodtheme": "Mood", "danceability_discogs_effnet": "Danceable", 
        "mood_happy_discogs_effnet": "Happy", "mood_sad_discogs_effnet": "Sad", 
        "mood_aggressive_discogs_effnet": "Aggressive", "mood_relaxed_discogs_effnet": "Relaxed", 
        "mood_party_discogs_effnet": "Party", "voice_instrumental_discogs_effnet": "Vocal", 
        "gender_discogs_effnet": "Gender"
    }

    for model_name in model_log_order:
        result = results.get(model_name)
        if not result: continue

        if "labels" in result and result["labels"]:
            summary_parts.append(f"{log_labels.get(model_name, model_name)}: {result['labels'][0].capitalize()}")
        elif "result" in result:
            positive_class = result['result']
            is_positive = not positive_class.startswith('not_') and not positive_class.startswith('non_')
            if model_name in ["gender_discogs_effnet", "voice_instrumental_discogs_effnet"]:
                 summary_parts.append(f"{log_labels.get(model_name, model_name)}: {positive_class.capitalize()}")
            else:
                 summary_parts.append(f"{log_labels.get(model_name, model_name)}: {'Yes' if is_positive else 'No'}")

    if not summary_parts:
        return "No conclusive tags found"
        
    return " | ".join(summary_parts)

def run_analysis(config: Dict):
    """Main function to run the entire analysis pipeline."""
    start_time = datetime.now()
    logging.info("="*20 + " Starting Universal Audio Analyzer " + "="*20)

    output_dir = Path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(config)
    if not models:
        logging.critical("Model loading failed. Exiting.")
        return

    all_audio_files = find_audio_files(config["input_directory"], config["file_pattern"])
    if not all_audio_files:
        return

    existing_stems = get_existing_json_stems(config["output_directory"])
    unprocessed_files = [f for f in all_audio_files if f.stem not in existing_stems]

    if not unprocessed_files:
        logging.info("All audio files already have corresponding JSON results. Nothing to do.")
        return

    max_files_to_process = config.get("max_files", 0)
    if max_files_to_process > 0:
        files_to_process = unprocessed_files[:max_files_to_process]
        logging.info(f"Processing the next batch of {len(files_to_process)} out of {len(unprocessed_files)} remaining files.")
    else:
        files_to_process = unprocessed_files
        logging.info(f"Processing all {len(unprocessed_files)} remaining files.")

    total_files = len(files_to_process)
    processed_count, error_count = 0, 0

    for i, audio_path in enumerate(files_to_process):
        file_start_time = datetime.now()
        result = analyze_audio_file(audio_path, models, config, output_dir)
        file_duration = (datetime.now() - file_start_time).total_seconds()

        if result and not result.get("error"):
            processed_count += 1
            summary = format_log_summary(result.get("analysis_results", {}))
            logging.info(f"[{i+1}/{total_files}] {audio_path.name} – ({summary}) (time: {file_duration:.2f}s)")
        else:
            error_count += 1
            logging.warning(f"[{i+1}/{total_files}] {audio_path.name} [ERROR] (time: {file_duration:.2f}s)")

    total_duration = (datetime.now() - start_time).total_seconds()
    logging.info("="*20 + " Analysis Complete " + "="*20)
    logging.info(f"Successfully processed: {processed_count} files")
    logging.info(f"Failed with errors: {error_count} files")
    logging.info(f"Total execution time: {timedelta(seconds=total_duration)}")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    setup_logging(CONFIG.get("loglevel", "INFO"))

    if not Path(CONFIG["input_directory"]).is_dir():
         logging.critical(f"Input directory does not exist: {CONFIG['input_directory']}")
         sys.exit(1)

    try:
        run_analysis(CONFIG)
    except Exception as e:
         logging.critical(f"A critical error occurred during execution: {e}")
         logging.critical(traceback.format_exc())
         sys.exit(1)
