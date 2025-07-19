#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Suppress TensorFlow INFO messages before importing essentia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import essentia.standard as es
import essentia
import matplotlib.pyplot as plt
import logging
import traceback
import subprocess
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Set

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
# --- Path Settings ---
INPUT_DIR = "music" # Example, please change
OUTPUT_DIR = "spectrograms"
PLOT_SUBDIR = "plots"
FEATURE_SUBDIR = "npz"
TEMP_DIR: Optional[str] = None
FFMPEG_EXECUTABLE: str = 'ffmpeg'

# --- Essentia Analysis Parameters ---
ESSENTIA_PARAMS: Dict[str, Any] = {
    'window_type': 'hann',
    'frame_size': 2048,
    'hop_size': 1024,
    'zero_padding': 2048,
    'num_mel_bands': 96,
    'mel_low_freq': 0.0,
    'mel_high_freq': 22050,
    'target_sample_rate': 44100,
    'log_bins_per_semitone': 1,
}

# --- Segment Analysis Parameters ---
ANALYSIS_START_TIME: float = 0
ANALYSIS_DURATION: float = 0

# --- File Settings ---
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif')
SAVE_DATA_TYPE: str = "all"
SPECTROGRAM_TYPES_TO_SAVE: List[str] = ["linear"]
FEATURE_FILE_SUFFIX = "_features.npz"

# --- Processing Control ---
MAX_FILES_TO_PROCESS: int = 3

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def extract_audio_segment_ffmpeg(
    input_path: Path,
    start_time: float,
    duration: float,
    target_sr: int,
    ffmpeg_exec: str,
    temp_dir: Optional[str] = None
) -> Optional[Path]:
    """
    Extracts an audio segment using FFmpeg, converts it to mono WAV, and resamples.
    
    Parameters:
    input_path (Path): Input audio file path.
    start_time (float): Start time of the segment in seconds.
    duration (float): Duration of the segment in seconds.
    target_sr (int): Target sample rate.
    ffmpeg_exec (str): FFmpeg executable path.
    temp_dir (Optional[str]): Temporary directory path (default: None).
    
    Returns:
    Optional[Path]: Path to the extracted audio segment or None on failure.
    """
    temp_wav_path: Optional[Path] = None
    resolved_temp_dir: Optional[Path] = None
    try:
        if temp_dir:
            resolved_temp_dir = Path(temp_dir).resolve()
            try:
                resolved_temp_dir.mkdir(parents=True, exist_ok=True)
                logging.debug(f"Using temporary directory: {resolved_temp_dir}")
            except OSError as e:
                logging.error(f"Could not create or access temporary directory {resolved_temp_dir}: {e}. Using system default.")
                resolved_temp_dir = None 

        fd, temp_wav_path_str = tempfile.mkstemp(
            suffix=".wav",
            prefix="segment_",
            dir=str(resolved_temp_dir) if resolved_temp_dir else None
        )
        os.close(fd)
        temp_wav_path = Path(temp_wav_path_str)
        logging.info(f"Preparing segment for '{input_path.name}' -> {temp_wav_path.name}")

        command = [
            ffmpeg_exec, '-hide_banner', '-loglevel', 'error',
            '-i', str(input_path.resolve()),
            '-ss', str(start_time),
            *(['-t', str(duration)] if duration > 0 else []),
            '-acodec', 'pcm_s16le',
            '-ar', str(target_sr),
            '-ac', '1',
            '-y',
            str(temp_wav_path.resolve())
        ]
        logging.debug(f"Executing ffmpeg command: {' '.join(command)}")

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            universal_newlines=True
        )

        if process.returncode != 0:
            logging.error(f"ffmpeg processing failed for '{input_path.name}' (code {process.returncode}).")
            if process.stderr: logging.error(f"ffmpeg stderr:\n{process.stderr.strip()}")
            if temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True)
            return None
        else:
            if temp_wav_path.exists() and temp_wav_path.stat().st_size > 0:
                logging.debug(f"ffmpeg processing successful: {temp_wav_path.name}")
                return temp_wav_path
            else:
                logging.error(f"ffmpeg reported success for '{input_path.name}' but output file is missing or empty: {temp_wav_path}")
                if process.stderr: logging.error(f"ffmpeg stderr (despite exit code 0):\n{process.stderr.strip()}")
                if temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True)
                return None
    except FileNotFoundError:
        logging.error(f"ffmpeg executable not found at '{ffmpeg_exec}'. Cannot process audio.")
        if temp_wav_path and temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True)
        return None
    except Exception as e:
        logging.error(f"Unexpected error during audio segment extraction for '{input_path.name}': {e}")
        logging.debug(traceback.format_exc())
        if temp_wav_path and temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True)
        return None


def generate_spectrograms(
    audio_path: Path,
    plot_output_dir: Path,
    feature_output_dir: Path,
    essentia_params: Dict[str, Any],
    analysis_params: Dict[str, Any],
    ffmpeg_exec: str,
    temp_dir: Optional[str],
    save_data_type: str, # Existing parameter
    spectrogram_types_to_save: List[str] # New parameter for specific types
):
    """
    Extracts audio segment, generates various spectrograms using Essentia,
    selectively saves plots and NPZ features based on configuration.
    """
    temp_wav_segment_path: Optional[Path] = None
    try:
        temp_wav_segment_path = extract_audio_segment_ffmpeg(
            input_path=audio_path,
            start_time=analysis_params['start_time'],
            duration=analysis_params['duration'],
            target_sr=essentia_params['target_sample_rate'],
            ffmpeg_exec=ffmpeg_exec,
            temp_dir=temp_dir
        )
        if temp_wav_segment_path is None:
            logging.warning(f"Skipping spectrogram generation for '{audio_path.name}' due to segment extraction failure.")
            return

        logging.info(f"Loading segment '{temp_wav_segment_path.name}' into Essentia...")
        loader = es.MonoLoader(filename=str(temp_wav_segment_path))
        audio = loader()
        sample_rate = loader.paramValue('sampleRate')
        if audio is None or len(audio) == 0:
            logging.warning(f"Essentia could not load or loaded empty audio from temporary file {temp_wav_segment_path.name}. Skipping.")
            return

        # Retrieve parameters from config
        frame_size = essentia_params['frame_size'] 
        hop_size = essentia_params['hop_size']
        common_window_type = essentia_params['window_type']
        common_zero_padding = essentia_params.get('zero_padding', 0)
        
        num_mel_bands = essentia_params['num_mel_bands']
        mel_low_freq = essentia_params['mel_low_freq']
        mel_high_freq = essentia_params['mel_high_freq']
        log_bins_per_semitone = essentia_params.get('log_bins_per_semitone', 1)

        # --- Determine which spectrograms to process and save based on new config ---
        # Normalize the list of types to save for case-insensitive and robust checking
        effective_types_lc = [s_type.lower().strip() for s_type in spectrogram_types_to_save]
        
        should_save_linear = "linear" in effective_types_lc
        should_save_mel = "mel" in effective_types_lc
        should_save_log = "log" in effective_types_lc
        
        # --- Essentia Algorithm Initialization ---
        frame_generator = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)
        
        windowing_algo = es.Windowing(type=common_window_type, zeroPadding=common_zero_padding)
        spectrum_algo = es.Spectrum() 

        fft_size_unified = frame_size + common_zero_padding 
        unified_input_spectrum_size = (fft_size_unified // 2) + 1

        melbands = es.MelBands(
            numberBands=num_mel_bands,
            lowFrequencyBound=mel_low_freq,
            highFrequencyBound=mel_high_freq,
            sampleRate=sample_rate,
            inputSize=unified_input_spectrum_size
        )
        
        spectrum_logfreq_algo = es.LogSpectrum(
            binsPerSemitone=log_bins_per_semitone, 
            sampleRate=sample_rate,
            frameSize=unified_input_spectrum_size
        )
        
        # Using the unified amp2db operator as per the latest version of user-provided code
        amp2db_operator = es.UnaryOperator(type='lin2db', scale=2.0) # Assuming scale=2.0 is the desired unified scaling

        # --- Frame Processing ---
        # These lists will be populated regardless, saving is conditional later
        spec_frames = []
        mel_frames = []
        logfreq_frames = []
        epsilon = 1e-9

        for frame in frame_generator:
            windowed_frame = windowing_algo(frame)
            frame_spectrum_mag = spectrum_algo(windowed_frame) # This is magnitude spectrum

            # Linear Spectrogram (from power)
            frame_spectrum_power = np.square(frame_spectrum_mag)
            spec_frames.append(amp2db_operator(frame_spectrum_power + epsilon))

            # Mel Spectrogram (from power)
            frame_mel_power = melbands(frame_spectrum_power) # melbands usually takes power spectrum
            mel_frames.append(amp2db_operator(frame_mel_power + epsilon))
            
            # Log-Frequency Spectrogram (from magnitude)
            frame_log_spectrum_mag_output, _, _ = spectrum_logfreq_algo(frame_spectrum_mag) # LogSpectrum takes magnitude
            logfreq_frames.append(amp2db_operator(frame_log_spectrum_mag_output + epsilon))


        if not spec_frames or not mel_frames or not logfreq_frames: # Basic check
             logging.warning(f"Could not extract sufficient features from segment '{temp_wav_segment_path.name}'. Skipping.")
             if temp_wav_segment_path and temp_wav_segment_path.exists(): temp_wav_segment_path.unlink(missing_ok=True)
             return

        # Convert lists of frames to numpy arrays and transpose
        linear_spectrogram_data = np.vstack(spec_frames).T
        mel_spectrogram_data = np.vstack(mel_frames).T
        log_frequency_spectrogram_data = np.vstack(logfreq_frames).T

        base_filename = audio_path.stem

        # --- Selective NPZ Saving ---
        # Governed by SAVE_DATA_TYPE. NPZ will now ALWAYS save only Mel data if SAVE_DATA_TYPE permits NPZ.
        # SPECTROGRAM_TYPES_TO_SAVE will NOT affect NPZ content for spectrogram types, only for plots.
        if save_data_type.lower() in ["all", "npz"]:
            feature_filename = feature_output_dir / f"{base_filename}{FEATURE_FILE_SUFFIX}"
            npz_payload = {} 
            
            # Always include metadata
            npz_payload['source_file'] = str(audio_path.resolve())
            npz_payload['sample_rate'] = sample_rate
            npz_payload['hop_size'] = hop_size
            npz_payload['mel_spectrogram'] = mel_spectrogram_data
            
            try:
                np.savez_compressed(feature_filename, **npz_payload)
                logging.info(f"Saved Features to NPZ Mel data: {feature_filename.name}")
            except Exception as e:
                logging.error(f"Failed to save NPZ features for {audio_path.name} to {feature_filename}: {e}")
                logging.debug(traceback.format_exc())
        else: 
            logging.info(f"Skipping NPZ feature saving for {audio_path.name} as per SAVE_DATA_TYPE ('{save_data_type}').")

        # --- Selective Plot Saving ---
        # Governed by SAVE_DATA_TYPE and further refined by SPECTROGRAM_TYPES_TO_SAVE
        if save_data_type.lower() in ["all", "spectrogram"]:
            # Define plot paths (even if not all are saved, for cleaner structure)
            plot_path_linear_spec = plot_output_dir / f"{base_filename}_linear_spectrogram.png"
            plot_path_mel = plot_output_dir / f"{base_filename}_mel_spectrogram.png"
            plot_path_logfreq = plot_output_dir / f"{base_filename}_logfreq_spectrogram.png"

            # Common plot elements
            num_time_frames = linear_spectrogram_data.shape[1] 
            time_axis_relative = np.arange(num_time_frames) * hop_size / sample_rate
            time_axis_absolute = time_axis_relative + analysis_params['start_time']
            time_start_abs = time_axis_absolute[0] if num_time_frames > 0 else analysis_params['start_time']
            time_end_abs = time_start_abs + (num_time_frames * hop_size / sample_rate) if num_time_frames > 0 else time_start_abs

            start_time_plot = analysis_params['start_time']
            duration_plot_config = analysis_params['duration']
            # Title segment should reflect the analyzed segment, not necessarily the plot's x-axis exact span if empty
            end_time_str_title = f"{start_time_plot + duration_plot_config:.1f}s" if duration_plot_config > 0 else "end"
            plot_title_segment = f" [{start_time_plot:.1f}s - {end_time_str_title}]"
            
            plots_saved_count = 0
            try:
                if should_save_linear:
                    fig_spec, ax_spec = plt.subplots(figsize=(24, 12))
                    im_spec = ax_spec.imshow(linear_spectrogram_data, aspect='auto', origin='lower', interpolation='none',
                                             extent=[time_start_abs, time_end_abs, 0, sample_rate / 2])
                    ax_spec.set_title(f"Linear Spectrogram (dB) - {audio_path.name}{plot_title_segment}")
                    ax_spec.set_ylabel("Frequency (Hz)")
                    ax_spec.set_xlabel(f"Time (s)")
                    fig_spec.colorbar(im_spec, ax=ax_spec, format='%+2.0f dB')
                    plt.tight_layout()
                    plt.savefig(plot_path_linear_spec)
                    plt.close(fig_spec)
                    logging.info(f"Saved Linear Spectrogram Plot: {plot_path_linear_spec.name}")
                    plots_saved_count +=1

                if should_save_mel:
                    fig_mel, ax_mel = plt.subplots(figsize=(24, 12))
                    im_mel = ax_mel.imshow(mel_spectrogram_data, aspect='auto', origin='lower', interpolation='none',
                                           extent=[time_start_abs, time_end_abs, 0, num_mel_bands])
                    ax_mel.set_title(f"Mel Spectrogram (dB) - {audio_path.name}{plot_title_segment}")
                    ax_mel.set_ylabel(f"Mel Bands ({num_mel_bands})")
                    ax_mel.set_xlabel(f"Time (s)")
                    fig_mel.colorbar(im_mel, ax=ax_mel, format='%+2.0f dB')
                    plt.tight_layout()
                    plt.savefig(plot_path_mel)
                    plt.close(fig_mel)
                    logging.info(f"Saved Mel Spectrogram Plot: {plot_path_mel.name}")
                    plots_saved_count += 1

                if should_save_log:
                    fig_logfreq, ax_logfreq = plt.subplots(figsize=(24, 12))
                    num_log_freq_bins = log_frequency_spectrogram_data.shape[0]
                    im_logfreq = ax_logfreq.imshow(log_frequency_spectrogram_data, aspect='auto', origin='lower', interpolation='none',
                                                   extent=[time_start_abs, time_end_abs, 0, num_log_freq_bins])
                    ax_logfreq.set_title(f"Log-Frequency Spectrogram (dB) - {audio_path.name}{plot_title_segment}")
                    ax_logfreq.set_ylabel(f"Log-Frequency Bins ({num_log_freq_bins})")
                    ax_logfreq.set_xlabel(f"Time (s)")
                    fig_logfreq.colorbar(im_logfreq, ax=ax_logfreq, format='%+2.0f dB')
                    plt.tight_layout()
                    plt.savefig(plot_path_logfreq)
                    plt.close(fig_logfreq)
                    logging.info(f"Saved Log-Frequency Spectrogram Plot: {plot_path_logfreq.name}")
                    plots_saved_count += 1
                
                if plots_saved_count == 0 and (should_save_linear or should_save_mel or should_save_log):
                    # This case should ideally not be hit if logic is correct, means no plots were requested by SPECTROGRAM_TYPES_TO_SAVE
                    # but SAVE_DATA_TYPE allowed plotting.
                    logging.info(f"No specific spectrogram plots were generated for {audio_path.name} based on SPECTROGRAM_TYPES_TO_SAVE, though SAVE_DATA_TYPE permitted plots.")
                elif not (should_save_linear or should_save_mel or should_save_log): # More explicit check if no types were ever true
                     logging.info(f"Skipping all plot saving for {audio_path.name}: No specific spectrogram types selected via SPECTROGRAM_TYPES_TO_SAVE.")


            except Exception as e:
                logging.error(f"Failed to create or save plot(s) for {audio_path.name}: {e}")
                logging.debug(traceback.format_exc())
        else:
            logging.info(f"Skipping all spectrogram plot saving for {audio_path.name} as per SAVE_DATA_TYPE ('{save_data_type}').")

    except Exception as e:
        logging.error(f"Major error during spectrogram generation for '{audio_path.name}': {e}")
        logging.debug(traceback.format_exc())

    finally:
        if temp_wav_segment_path and temp_wav_segment_path.exists():
            try:
                temp_wav_segment_path.unlink()
                logging.debug(f"Cleaned up temporary file: {temp_wav_segment_path.name}")
            except OSError as e:
                logging.error(f"Error deleting temporary file {temp_wav_segment_path}: {e}")


def main():
    script_dir = Path(__file__).parent.resolve()
    logging.info(f"Script directory: {script_dir}")

    # --- Load Configuration ---
    input_dir_config = INPUT_DIR
    output_dir_config = OUTPUT_DIR
    plot_subdir_config = PLOT_SUBDIR
    feature_subdir_config = FEATURE_SUBDIR
    temp_dir_config = TEMP_DIR
    ffmpeg_executable_config = FFMPEG_EXECUTABLE
    essentia_params_config = ESSENTIA_PARAMS
    analysis_params_config = {
        'start_time': ANALYSIS_START_TIME,
        'duration': ANALYSIS_DURATION
    }
    supported_extensions_config = SUPPORTED_EXTENSIONS
    max_files_to_process_config = MAX_FILES_TO_PROCESS
    save_data_type_config = SAVE_DATA_TYPE
    # Load new configuration for selecting which spectrogram types to save
    spectrogram_types_to_save_config = SPECTROGRAM_TYPES_TO_SAVE 

    # --- Resolve Paths ---
    input_directory = Path(input_dir_config)
    output_directory = Path(output_dir_config)
    if not input_directory.is_absolute():
         input_directory = (script_dir / input_directory).resolve()
    if not output_directory.is_absolute():
         output_directory = (script_dir / output_directory).resolve()

    plot_output_dir = output_directory / plot_subdir_config
    feature_output_dir = output_directory / feature_subdir_config

    # --- Validate and Create Directories ---
    if not input_directory.is_dir():
        logging.error(f"Input directory '{input_directory}' not found or is not a directory.")
        return
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using input directory: {input_directory}")
        logging.info(f"Plot images will be saved to: {plot_output_dir}")
        logging.info(f"Feature files (.npz) will be saved to: {feature_output_dir}")
    except OSError as e:
         logging.error(f"Could not create output directories under '{output_directory}': {e}")
         return

    # --- Log Configurations ---
    if temp_dir_config: logging.info(f"Using temporary directory: {temp_dir_config}")
    else: logging.info("Using system default temporary directory.")
    logging.info(f"Using ffmpeg executable: {ffmpeg_executable_config}")
    duration_log = 'Entire file' if analysis_params_config['duration'] <= 0 else f"{analysis_params_config['duration']}s"
    logging.info(f"Analysis segment: Start={analysis_params_config['start_time']}s, Duration={duration_log}")
    logging.info(f"Looking for files with extensions: {supported_extensions_config}")
    logging.info(f"Data saving type (SAVE_DATA_TYPE): {save_data_type_config}")
    # Log the new configuration setting
    logging.info(f"Spectrogram types to save (SPECTROGRAM_TYPES_TO_SAVE): {spectrogram_types_to_save_config}") 
    if max_files_to_process_config > 0: logging.info(f"Processing limit: Max {max_files_to_process_config} new files.")
    else: logging.info("Processing limit: All new files will be processed.")
    logging.info(f"Essentia parameters: {essentia_params_config}")


    if not shutil.which(ffmpeg_executable_config):
         logging.error(f"ffmpeg executable '{ffmpeg_executable_config}' not found. Please install or correct the path.")
         return

    # --- File Processing Logic (including blacklist) ---
    # Blacklist logic depends on NPZ files being potentially saved.
    # If NPZ saving is generally disabled by SAVE_DATA_TYPE, or if SPECTROGRAM_TYPES_TO_SAVE results in no NPZ data,
    # the blacklist will not be effectively updated for those files. This is acceptable.
    processed_file_stems: Set[str] = set()
    if feature_output_dir.exists() and save_data_type_config.lower() in ["all", "npz"]:
        # Only build blacklist if NPZ files could be written based on SAVE_DATA_TYPE
        # The effectiveness of blacklist also depends on SPECTROGRAM_TYPES_TO_SAVE allowing some NPZ content
        for npz_file in feature_output_dir.glob(f"*{FEATURE_FILE_SUFFIX}"): # Use FEATURE_FILE_SUFFIX from config
            base_name = npz_file.name[:-len(FEATURE_FILE_SUFFIX)]
            processed_file_stems.add(base_name)

    all_input_audio_files: List[Path] = [
        item_path for item_path in input_directory.rglob('*') 
        if item_path.is_file() and item_path.suffix.lower() in supported_extensions_config
    ]

    files_to_process_after_blacklist: List[Path] = []
    skipped_count = 0
    # Apply blacklist filtering only if NPZs are potentially being saved.
    # The decision to save an NPZ for a specific file also depends on SPECTROGRAM_TYPES_TO_SAVE.
    if save_data_type_config.lower() in ["all", "npz"]:
        for audio_file in all_input_audio_files:
            if audio_file.stem not in processed_file_stems:
                files_to_process_after_blacklist.append(audio_file)
            else:
                logging.debug(f"Skipping already processed (NPZ found): {audio_file.name}")
                skipped_count += 1
    else: 
        files_to_process_after_blacklist = all_input_audio_files
        logging.info("NPZ saving is disabled by SAVE_DATA_TYPE; blacklist based on existing NPZ files is not applied.")
    
    logging.info("--- File Scan and Filtering Summary ---")
    logging.info(f"Total audio files found: {len(all_input_audio_files)}")
    if save_data_type_config.lower() in ["all", "npz"] and processed_file_stems: # Log blacklist info only if relevant
        logging.info(f"Blacklist entries (from existing NPZs): {len(processed_file_stems)}")
    logging.info(f"Files skipped (due to blacklist): {skipped_count}")
    logging.info(f"Files available post-blacklist: {len(files_to_process_after_blacklist)}")
    
    files_to_process_this_run: List[Path]
    if max_files_to_process_config > 0 and len(files_to_process_after_blacklist) > max_files_to_process_config:
        files_to_process_this_run = files_to_process_after_blacklist[:max_files_to_process_config]
        logging.info(f"Applying MAX_FILES_TO_PROCESS: selecting first {max_files_to_process_config} of {len(files_to_process_after_blacklist)}.")
    else:
        files_to_process_this_run = files_to_process_after_blacklist
        if max_files_to_process_config > 0 : logging.info(f"MAX_FILES_TO_PROCESS ({max_files_to_process_config}) not exceeded by available files.")

    logging.info(f"Files to be processed in this run: {len(files_to_process_this_run)}")
    logging.info("-------------------------------------")

    if not files_to_process_this_run:
        logging.info("No files to process in this run.")
        logging.info(f"\n--- Processing complete ---")
        return

    # --- Iterate and Process Files ---
    logging.info(f"Starting processing of {len(files_to_process_this_run)} file(s)...")
    session_processed_count = 0
    session_error_count = 0

    for i, item in enumerate(files_to_process_this_run):
        logging.info(f"--- Processing file {i+1}/{len(files_to_process_this_run)}: {item.name} ---")
        try:
            generate_spectrograms(
                item,
                plot_output_dir,
                feature_output_dir,
                essentia_params_config,
                analysis_params_config,
                ffmpeg_executable_config,
                temp_dir_config,
                save_data_type_config,
                spectrogram_types_to_save_config # Pass the new config
            )
            session_processed_count += 1
        except Exception as e: 
            logging.error(f"Critical error processing file {item.name} in main loop: {e}")
            logging.debug(traceback.format_exc())
            session_error_count += 1
    
    # --- Final Summary ---
    logging.info(f"\n--- Processing complete ---")
    logging.info(f"Total audio files found: {len(all_input_audio_files)}")
    logging.info(f"Files skipped (due to blacklist): {skipped_count}")
    logging.info(f"Attempted to process in this run: {len(files_to_process_this_run)}")
    logging.info(f"Successfully processed files in this run: {session_processed_count}")
    logging.info(f"Errors during processing in this run: {session_error_count}")

if __name__ == "__main__":
    main()