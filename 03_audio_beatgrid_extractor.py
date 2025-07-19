#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Suppress TensorFlow INFO messages before importing essentia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import essentia.standard as es
import glob
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
MODEL_DIR = 'models'
AUDIO_DIR = 'music'
# Directory for output text files (BPM data)
OUTPUT_DATA_DIR = 'output_data'
# Directory for plots, will be a subfolder named 'output_plots' within OUTPUT_DATA_DIR
OUTPUT_PLOT_DIR = os.path.join(OUTPUT_DATA_DIR, 'output_plots')

# Model .pb filename (ensure it matches your file in the models/ directory)
MODEL_FILENAME_PB = 'deeptemp-k16-3.pb'

# Sample rate the TempoCNN model was trained on (from the example)
TARGET_SR = 11025

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.aif', '.aiff')

# Duration of audio to display on the plot (in seconds)
PLOT_DURATION_SEC = 60

# DPI (Dots Per Inch) for saved plots. Higher DPI means higher resolution.
PLOT_DPI = 900

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_model_path(model_dir, model_filename_pb):
    """Finds the path to the .pb model file."""
    pb_path = os.path.join(model_dir, model_filename_pb)
    if not os.path.exists(pb_path):
        print(f"Error: Model .pb file not found at: {pb_path}")
        return None
    return pb_path

def analyze_audio_with_tempocnn(audio_filepath, model_graph_path):
    """Analyzes a single audio file to get global BPM, local BPMs, and local probabilities using TempoCNN.
    Returns global_bpm, local_bpm, local_probs, audio_data (resampled), and actual sample_rate.
    """
    try:
        # Load audio, resample to the target sample rate for the model
        loader = es.MonoLoader(filename=audio_filepath, sampleRate=TARGET_SR)
        audio = loader()

        if audio.size == 0:
            print(f"Warning: Could not load or empty audio file: {audio_filepath}")
            return None, None, None, None, None

        # Initialize and run TempoCNN
        tempo_cnn_algo = es.TempoCNN(graphFilename=model_graph_path)
        global_bpm, local_bpm, local_probs = tempo_cnn_algo(audio)

        return global_bpm, local_bpm, local_probs, audio, TARGET_SR

    except Exception as e:
        print(f"Error analyzing file {audio_filepath}: {e}")
        return None, None, None, None, None

def plot_and_save_beat_grid(audio_data, sample_rate, global_bpm, output_plot_filepath, duration_sec):
    """Plots the waveform with a beat grid and saves it to a file with specified DPI."""
    try:
        plt.figure(figsize=(15, 5)) # Width, Height in inches
        plot_samples = int(sample_rate * duration_sec)
        audio_slice = audio_data[:plot_samples]
        
        if len(audio_slice) == 0:
            print(f"Warning: Audio slice is empty for plotting for {os.path.basename(output_plot_filepath)}. Skipping plot.")
            plt.close()
            return

        time_axis = np.arange(0, len(audio_slice)) / sample_rate

        plt.plot(time_axis, audio_slice, label="Waveform")

        track_name_for_plot = os.path.basename(output_plot_filepath).replace('_beatgrid.png', '')

        if global_bpm and global_bpm > 0:
            interval_seconds = 60.0 / global_bpm
            beat_markers_time = np.arange(0, duration_sec + interval_seconds, interval_seconds) 

            for marker_time in beat_markers_time:
                if time_axis.size > 0 and marker_time <= time_axis[-1]:
                     plt.axvline(x=marker_time, color='red', linestyle='--', alpha=0.7)
                elif time_axis.size == 0 and marker_time <= duration_sec:
                     plt.axvline(x=marker_time, color='red', linestyle='--', alpha=0.7)

            plt.title(f"Waveform and Beat Grid (BPM: {global_bpm:.2f}) for {track_name_for_plot}")
        else:
            plt.title(f"Waveform (BPM not determined) for {track_name_for_plot}")

        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout() # Adjusts plot to prevent labels from being cut off
        plt.savefig(output_plot_filepath, dpi=PLOT_DPI) # Use the configured DPI
        plt.close() 
        print(f"Plot saved: {output_plot_filepath} (DPI: {PLOT_DPI})")

    except Exception as e:
        print(f"Error creating plot for BPM {global_bpm} (file {os.path.basename(output_plot_filepath)}): {e}")
        if plt.gcf().get_axes():
            plt.close()


def save_bpm_data_to_text_file(original_filename, global_bpm, local_bpm, local_probs, output_dir):
    """Saves the BPM data (global, local, and probabilities) to a text file for the given track."""
    base_filename = os.path.splitext(original_filename)[0]
    txt_filename = f"{base_filename}_bpm_data.txt"
    txt_filepath = os.path.join(output_dir, txt_filename)

    try:
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"File: {original_filename}\n")
            if global_bpm is not None:
                f.write(f"Global BPM: {global_bpm:.2f}\n\n")
                
                f.write("Local BPM Estimates:\n")
                if local_bpm is not None and local_bpm.size > 0:
                    for i, bpm_val in enumerate(local_bpm):
                        prob_val_str = f"{local_probs[i]:.4f}" if local_probs is not None and i < len(local_probs) else "N/A"
                        f.write(f"  Segment {i+1}: {bpm_val:.2f} BPM (Probability: {prob_val_str})\n")
                else:
                    f.write("  No local BPM estimates available.\n")
            else:
                f.write("Error: Could not determine BPM.\n")
        print(f"BPM data saved to: {txt_filepath}")
    except IOError:
        print(f"Error: Could not write text file to {txt_filepath}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("Starting BPM analysis script with TempoCNN...")

    # 1. Find model path
    model_pb_path = find_model_path(MODEL_DIR, MODEL_FILENAME_PB)
    if not model_pb_path:
        return

    print(f"Model found: {model_pb_path}")

    # 2. Create output directories if they don't exist
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    print(f"Output directory for BPM text files: {os.path.abspath(OUTPUT_DATA_DIR)}")
    print(f"Output directory for plots: {os.path.abspath(OUTPUT_PLOT_DIR)}")

    # 3. Find audio files
    audio_files = []
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, '**', f"*{ext}"), recursive=True))
    
    if not audio_files:
        print(f"No audio files found in directory {AUDIO_DIR} or its subdirectories.")
        return

    print(f"Found {len(audio_files)} audio files for analysis.")

    # 4. Process each audio file
    for audio_filepath in audio_files:
        print(f"\nAnalyzing file: {audio_filepath}...")
        
        filename_with_ext = os.path.basename(audio_filepath)
        global_bpm, local_bpm, local_probs, audio_for_plot, sr_for_plot = analyze_audio_with_tempocnn(audio_filepath, model_pb_path)

        save_bpm_data_to_text_file(filename_with_ext, global_bpm, local_bpm, local_probs, OUTPUT_DATA_DIR)

        if global_bpm is not None and audio_for_plot is not None:
            print(f"Determined Global BPM: {global_bpm:.2f}")

            plot_filename_base = os.path.splitext(filename_with_ext)[0]
            output_plot_filepath = os.path.join(OUTPUT_PLOT_DIR, f"{plot_filename_base}_beatgrid.png")
            plot_and_save_beat_grid(audio_for_plot, sr_for_plot, global_bpm, output_plot_filepath, PLOT_DURATION_SEC)
        else:
            print(f"Could not fully process file: {filename_with_ext}")

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()