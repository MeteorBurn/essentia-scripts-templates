#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Music Feature Extractor:
Extracts detailed audio features using Essentia MusicExtractor.
Analyzes audio files in the specified directory, optionally converts non-WAV files
using ffmpeg, and saves results to individual JSON files for each track.
Implements a blacklist to avoid re-processing files with existing JSON results.
"""

import os

# Suppress TensorFlow INFO messages before importing essentia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import traceback
from typing import List, Dict, Optional, Tuple, Any, Union, Set
import subprocess
import shutil
import tempfile
import sys
import argparse

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
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO, # Default level, can be overridden later
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
CONFIG = {
    # --- Input/Output ---
    "input_directory": "music", # Source audio directory
    "output_directory": "results", # Directory for JSON output results
    "file_pattern": "",             # Optional: File filter by pattern (e.g., "track_")
    "max_files": 50,                 # Max number of files to process (0 for all)

    # --- Conversion Parameters ---
    "convert_to_wav": False,        # Convert non-WAV to temporary WAV before analysis
    "ffmpeg_path": "ffmpeg",        # Path to ffmpeg executable
    "temp_dir": "",                 # Optional: Directory for temporary WAV files (uses system default if empty)

    # --- Essentia Parameters ---
    "sample_rate": 44100,           # Target sample rate for Essentia
    "resample_quality": 4,          # Resampling quality (0=low, 4=high)
    "essentia_stats": ['mean', 'stdev'], # 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2', 'var'
}

# =============================================================================
# CONSTANTS
# =============================================================================
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif', '.m4a']
# Type hint for combined JSON result
FeatureAnalysisResult = Dict[str, Any]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def find_audio_files(directory: Union[str, Path], file_pattern: str = "", max_files: int = 0) -> List[Path]:
    audio_files = []
    start_dir = Path(directory)
    pattern_lower = file_pattern.lower() if file_pattern else None
    if not start_dir.is_dir():
        logging.error(f"Input directory not found or is not a directory: {start_dir}")
        return []
    logging.info(f"Searching for audio files with extensions {AUDIO_EXTENSIONS} in {start_dir}...")
    count = 0
    for item in start_dir.rglob('*'):
        if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
            if pattern_lower:
                if pattern_lower not in item.stem.lower():
                    continue
            audio_files.append(item)
            count += 1
            if max_files > 0 and count >= max_files:
                logging.info(f"Reached max_files limit ({max_files}).")
                break
    logging.info(f"Found {len(audio_files)} audio files matching criteria.")
    return audio_files

def get_existing_json_stems(directory: Union[str, Path]) -> Set[str]:
    json_stems = set()
    output_dir = Path(directory)
    if not output_dir.is_dir():
        logging.warning(f"Output directory for JSON check not found: {output_dir}. Assuming no existing files.")
        return json_stems

    logging.info(f"Checking for existing JSON files in: {output_dir.resolve()}")
    for item in output_dir.glob('*.json'):
        if item.is_file():
            json_stems.add(item.stem)
    logging.info(f"Found {len(json_stems)} existing JSON files.")
    return json_stems

def pool_to_dict(pool: essentia.Pool) -> Dict[str, Any]:
    data = {}
    for desc_name in pool.descriptorNames():
        value = pool[desc_name]
        # Convert numpy arrays/scalars to lists/native types
        if isinstance(value, np.ndarray):
            data[desc_name] = value.tolist()
        elif isinstance(value, np.generic):
            data[desc_name] = value.item()
        elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
             data[desc_name] = value
        else:
            # Fallback for other types
            data[desc_name] = str(value)
            logging.debug(f"Converted descriptor '{desc_name}' of type {type(value)} to string as fallback.")

    # Reorganize nested structures
    organized_data = {}
    for key, value in data.items():
        parts = key.split('.')
        d = organized_data
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                d[part] = value
            else:
                d = d.setdefault(part, {})
    return organized_data

# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================
def convert_audio_to_wav(input_path: Path, target_sr: int, ffmpeg_exec: str, temp_dir: Optional[str]) -> Optional[Path]:
    temp_wav_path: Optional[Path] = None
    try:
        resolved_temp_dir = None
        if temp_dir:
            resolved_temp_dir = Path(temp_dir)
            try:
                resolved_temp_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logging.error(f"Could not create temporary directory {resolved_temp_dir}: {e}. Using system default.")
                resolved_temp_dir = None
        fd, temp_wav_path_str = tempfile.mkstemp(suffix=".wav", prefix="convert_", dir=str(resolved_temp_dir) if resolved_temp_dir else None)
        os.close(fd)
        temp_wav_path = Path(temp_wav_path_str)
        logging.info(f"Converting '{input_path.name}' to temporary WAV: {temp_wav_path.name}")
        command = [
            ffmpeg_exec, '-hide_banner', '-loglevel', 'error',
            '-i', str(input_path.resolve()), '-y',
            '-acodec', 'pcm_s16le', '-ar', str(target_sr), '-ac', '1',
            str(temp_wav_path.resolve())
        ]
        logging.debug(f"Executing ffmpeg command: {' '.join(command)}")
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, universal_newlines=True)
        if process.returncode != 0:
            logging.error(f"ffmpeg conversion failed for {input_path.name} (code {process.returncode}).")
            if process.stderr: logging.error(f"ffmpeg stderr:\n{process.stderr.strip()}")
            if temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True)
            return None
        else:
            if temp_wav_path.exists() and temp_wav_path.stat().st_size > 0:
                logging.debug(f"ffmpeg conversion successful: {temp_wav_path.name}")
                return temp_wav_path
            else:
                logging.error(f"ffmpeg reported success but output file is missing/empty: {temp_wav_path}")
                if temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True)
                return None
    except FileNotFoundError:
        logging.error(f"ffmpeg executable not found at '{ffmpeg_exec}'. Cannot convert.")
        return None
    except Exception as e:
        logging.error(f"Error during audio conversion for {input_path.name}: {e}")
        logging.debug(traceback.format_exc())
        if temp_wav_path and temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)
        return None

# =============================================================================
# CORE ANALYSIS FUNCTION
# =============================================================================
def extract_audio_features(
    original_audio_path: Path,
    config: Dict,
    output_dir: Path
) -> Optional[FeatureAnalysisResult]:
    """
    Extracts music features using Essentia MusicExtractor.
    """
    path_to_analyze = original_audio_path
    temp_wav_file: Optional[Path] = None
    needs_cleanup = False

    # --- Optional Conversion ---
    if config["convert_to_wav"] and original_audio_path.suffix.lower() != '.wav':
        temp_wav_file = convert_audio_to_wav(
            original_audio_path, config["sample_rate"], config["ffmpeg_path"], config.get("temp_dir") or None
        )
        if temp_wav_file and temp_wav_file.exists():
            path_to_analyze = temp_wav_file
            needs_cleanup = True
            logging.debug(f"Using temporary file for analysis: {path_to_analyze.name}")
        else:
            logging.warning(f"Skipping analysis for {original_audio_path.name} due to conversion failure.")
            if temp_wav_file and not temp_wav_file.exists(): needs_cleanup = False
            return None

    # --- Initialize JSON Data Structure ---
    json_data: FeatureAnalysisResult = {
        "file_path_original": str(original_audio_path.resolve()),
        "file_path_analyzed": str(path_to_analyze.resolve()),
        "filename": original_audio_path.name,
        "analysis_timestamp": datetime.now().isoformat(),
        "analysis_config": {
             "sample_rate": config["sample_rate"],
             "essentia_stats_computed": config.get("essentia_stats", []),
        },
        "essentia_features": {}
    }

    try:
        # --- Essentia Feature Extraction (MusicExtractor) ---
        logging.info("Running Essentia MusicExtractor...")
        try:
            stats = config.get("essentia_stats", ['mean', 'stdev'])
            extractor = es.MusicExtractor(lowlevelStats=stats, rhythmStats=stats, tonalStats=stats)
            features_pool, _ = extractor(str(path_to_analyze))
            logging.info(f"  Essentia extraction successful. Found {len(features_pool.descriptorNames())} descriptors.")
            json_data["essentia_features"] = pool_to_dict(features_pool)
        except essentia.EssentiaException as ess_err:
            logging.error(f"Error during Essentia MusicExtractor analysis for {path_to_analyze.name}: {ess_err}")
            json_data["essentia_features"] = {"error": str(ess_err)}
        except Exception as ext_err:
             logging.error(f"Unexpected error during Essentia MusicExtractor analysis: {ext_err}")
             json_data["essentia_features"] = {"error": str(ext_err)}

        # --- Save JSON File ---
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            json_filename = output_dir / (original_audio_path.stem + ".json")
            logging.debug(f"Saving JSON results to: {json_filename}")
            with open(json_filename, 'w', encoding='utf-8') as f_json:
                json.dump(json_data, f_json, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        except OSError as json_dir_err:
            logging.error(f"Could not write to JSON output directory {output_dir}: {json_dir_err}")
        except Exception as json_save_err:
            logging.error(f"Could not save JSON file {json_filename}: {json_save_err}")
            logging.debug(traceback.format_exc())

        return json_data

    except RuntimeError as es_error:
         logging.error(f"Essentia runtime error analyzing {path_to_analyze.name}: {es_error}")
         return None
    except Exception as e:
        logging.error(f"Unexpected error during analysis pipeline for {original_audio_path.name}: {e}")
        logging.debug(traceback.format_exc())
        return None
    finally:
        # Cleanup temp file
        if needs_cleanup and temp_wav_file and temp_wav_file.exists():
            try:
                logging.debug(f"Deleting temporary file: {temp_wav_file}")
                temp_wav_file.unlink(missing_ok=True)
            except OSError as e_del:
                logging.warning(f"Could not delete temporary file {temp_wav_file}: {e_del}")

# =============================================================================
# Custom JSON Encoder for Numpy Types
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            if np.isnan(obj): return None
            if np.isinf(obj): return None
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return [self.default(item) for item in obj.tolist()]
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_feature_extraction(config: Dict):
    start_time = datetime.now()
    logging.info("="*20 + " Starting Music Feature Extractor " + "="*20)
    if config["convert_to_wav"]:
        ffmpeg_path = config.get("ffmpeg_path", "ffmpeg")
        resolved_ffmpeg_path = shutil.which(ffmpeg_path)
        if resolved_ffmpeg_path:
            logging.info(f"ffmpeg found at: {resolved_ffmpeg_path}")
        else:
            logging.error(f"ffmpeg command '{ffmpeg_path}' not found.")
            logging.error("Audio conversion requires ffmpeg. Install it or set 'convert_to_wav' to False.")
            return
            
    input_dir = Path(config["input_directory"])
    output_dir = Path(config["output_directory"])
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output JSON files will be saved directly in: {output_dir.resolve()}")
    except OSError as e:
        logging.error(f"Could not create output directory {output_dir}: {e}")
        return

    all_audio_files = find_audio_files(input_dir, config["file_pattern"], config["max_files"])
    if not all_audio_files:
        logging.warning("No matching audio files found in input directory. Exiting.")
        return
        
    existing_json_stems = get_existing_json_stems(output_dir)
    all_audio_stems = {audio_file.stem for audio_file in all_audio_files}
    stems_to_process = all_audio_stems - existing_json_stems
    
    logging.info(f"Found {len(all_audio_stems)} total audio files.")
    logging.info(f"Found {len(existing_json_stems)} existing JSON results.")
    logging.info(f"{len(stems_to_process)} audio files need processing.")
    
    audio_files_to_process = [
        audio_file for audio_file in all_audio_files if audio_file.stem in stems_to_process
    ]
    
    total_files_to_process = len(audio_files_to_process)
    if total_files_to_process == 0:
        logging.info("All found audio files already have corresponding JSON results. Nothing to process. Exiting.")
        return
        
    processed_count = 0
    skipped_count = 0
    
    logging.info(f"Starting feature extraction of {total_files_to_process} files...")
    for i, audio_path in enumerate(audio_files_to_process):
        file_start_time = datetime.now()
        logging.info(f"--- Processing [{i+1}/{total_files_to_process}]: {audio_path.name} ---")
        
        analysis_result_json = extract_audio_features(audio_path, config, output_dir)
        
        file_end_time = datetime.now()
        file_duration = (file_end_time - file_start_time).total_seconds()
        
        if analysis_result_json is not None:
            has_essentia_error = isinstance(analysis_result_json.get("essentia_features", {}), dict) and "error" in analysis_result_json["essentia_features"]
            
            if has_essentia_error:
                 skipped_count += 1
                 logging.warning(f"  Completed processing for {audio_path.name} with internal analysis errors (time: {file_duration:.2f}s). Check JSON for details.")
            else:
                 processed_count += 1
                 logging.info(f"  Successfully processed and saved JSON in {file_duration:.2f}s.")
        else:
            skipped_count += 1
            logging.warning(f"  Skipped file {audio_path.name} due to critical errors (e.g., load/convert failed) (time: {file_duration:.2f}s).")
            
    end_time = datetime.now()
    total_duration_seconds = (end_time - start_time).total_seconds()
    total_duration_td = timedelta(seconds=total_duration_seconds)
    
    logging.info("="*20 + " Feature Extraction Complete " + "="*20)
    logging.info(f"Attempted to process: {total_files_to_process} files")
    logging.info(f"Successfully processed and saved JSON for: {processed_count} files")
    logging.info(f"Skipped due to errors (or completed with errors): {skipped_count} files")
    logging.info(f"Files ignored due to existing JSON: {len(all_audio_files) - total_files_to_process}")
    logging.info(f"Total execution time: {total_duration_seconds:.2f} seconds ({str(total_duration_td).split('.')[0]})")
    
    if processed_count > 0:
        logging.info(f"Average time per successfully processed file: {total_duration_seconds / processed_count:.2f} seconds")
    
    logging.info(f"Individual JSON feature results saved in: {output_dir.resolve()}")
    logging.info("="*20 + " Script Finished " + "="*20)

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser_main = argparse.ArgumentParser(description="Music Feature Extractor Script")
    parser_main.add_argument("-i", "--input", help="Override input directory from CONFIG")
    parser_main.add_argument("-o", "--output", help="Override output directory from CONFIG")
    parser_main.add_argument("-l", "--loglevel", help="Set logging level (DEBUG, INFO, WARNING, ERROR)", default="INFO")
    
    args_main = parser_main.parse_args()
    
    log_level_str = args_main.loglevel.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.info(f"Logging level set to: {log_level_str}")
    
    config_updated = False
    
    if args_main.input:
        if Path(args_main.input).is_dir():
            CONFIG["input_directory"] = args_main.input
            logging.info(f"Overriding input directory with: {args_main.input}")
            config_updated = True
        else:
            logging.error(f"Input directory specified via --input does not exist: {args_main.input}. Using CONFIG value.")
            
    if args_main.output:
        CONFIG["output_directory"] = args_main.output
        logging.info(f"Overriding output directory with: {args_main.output}")
        config_updated = True
        
    if not Path(CONFIG["input_directory"]).is_dir():
         logging.critical(f"Input directory specified does not exist: {CONFIG['input_directory']}")
         print(f"\nCRITICAL ERROR: Input directory not found: {CONFIG['input_directory']}", file=sys.stderr)
         sys.exit(1)
         
    try:
        run_feature_extraction(CONFIG)
    except Exception as main_err:
         logging.critical(f"A critical error occurred during script execution: {main_err}")
         logging.critical(traceback.format_exc())
         print(f"\nCRITICAL ERROR: The script terminated unexpectedly: {main_err}. Please check the log.", file=sys.stderr)
         sys.exit(1)