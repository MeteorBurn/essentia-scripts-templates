#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
from pathlib import Path
import logging

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
CONFIG = {
    "json_directory": "avantgarde_results",
    "output_excel_path": "avantgarde_analysis.xlsx",
    "loglevel": "INFO"
}

# =============================================================================
# CONSTANTS
# =============================================================================
BROKEN_BEAT_GENRES = [
    "Breakbeat", "Breakcore", "Breaks", "Progressive Breaks", "Broken Beat", 
    "Drum n Bass", "Jungle", "Halftime", "Juke", 
    "UK Garage", "Speed Garage", "Bassline", "Electro"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def setup_logging(level: str):
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def find_json_files(directory: str):
    json_files = []
    start_dir = Path(directory)
    if not start_dir.is_dir():
        logging.error(f"Directory not found: {start_dir}")
        return []
    
    for item in start_dir.rglob('*.json'):
        if item.is_file():
            json_files.append(item)
    
    logging.info(f"Found {len(json_files)} JSON files")
    return sorted(json_files)

def check_broken_beat(maest_genres, discogs_genres):
    all_genres = maest_genres + discogs_genres
    return any(genre in BROKEN_BEAT_GENRES for genre in all_genres)

def safe_get_labels_confidences(data, key):
    result = data.get(key, {})
    labels = result.get('labels', [])
    confidences = result.get('confidences', [])
    return labels, confidences

def format_list_with_comma(items):
    if not items:
        return ""
    # Удаляем дубликаты, сохраняя порядок
    unique_items = []
    seen = set()
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return ", ".join(unique_items)

def format_confidences(confidences):
    return ", ".join([str(c) for c in confidences]) if confidences else ""

def process_json_file(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis_results = data.get('analysis_results', {})
        
        maest_labels, maest_confidences = safe_get_labels_confidences(analysis_results, 'maest_400_v2')
        discogs_labels, discogs_confidences = safe_get_labels_confidences(analysis_results, 'genre_discogs400')
        mood_labels, mood_confidences = safe_get_labels_confidences(analysis_results, 'mtg_jamendo_moodtheme')
        instrument_labels, instrument_confidences = safe_get_labels_confidences(analysis_results, 'mtg_jamendo_instrument')
        tags_labels, tags_confidences = safe_get_labels_confidences(analysis_results, 'mtt_discogs_effnet')
        
        file_path_data = data.get('file_path', {})
        win_path = file_path_data.get('win', '')
        
        is_broken = check_broken_beat(maest_labels, discogs_labels)
        
        return {
            'file_path': win_path,
            'file_name': data.get('file_name', ''),
            'genres_maest_400_v2': format_list_with_comma(maest_labels),
            'confidences': format_confidences(maest_confidences),
            'genre_discogs400': format_list_with_comma(discogs_labels),
            'confidences.1': format_confidences(discogs_confidences),
            'is_broken_beat': is_broken,
            'jamendo_mood': format_list_with_comma(mood_labels),
            'confidences.2': format_confidences(mood_confidences),
            'jamendo_instrument': format_list_with_comma(instrument_labels),
            'confidences.3': format_confidences(instrument_confidences),
            'discogs_tags': format_list_with_comma(tags_labels),
            'confidences.4': format_confidences(tags_confidences)
        }
        
    except Exception as e:
        logging.error(f"Error processing {json_path}: {e}")
        return None

def load_existing_excel(excel_path):
    """Загружает существующие данные из Excel файла"""
    try:
        if Path(excel_path).exists():
            df = pd.read_excel(excel_path, engine='openpyxl')
            logging.info(f"Loaded existing Excel file with {len(df)} rows")
            return df
        else:
            logging.info("Excel file does not exist, will create new one")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading existing Excel file: {e}")
        return pd.DataFrame()

def filter_new_records(new_data, existing_df):
    """Фильтрует новые записи, исключая дубликаты по file_path"""
    if existing_df.empty:
        return new_data
    
    # Получаем список существующих путей к файлам
    existing_paths = set(existing_df['file_path'].tolist()) if 'file_path' in existing_df.columns else set()
    
    # Фильтруем только новые записи
    new_records = []
    duplicates_count = 0
    
    for record in new_data:
        if record['file_path'] not in existing_paths:
            new_records.append(record)
        else:
            duplicates_count += 1
    
    logging.info(f"Found {len(new_records)} new records, {duplicates_count} duplicates skipped")
    return new_records

def create_excel_report(config):
    # Находим JSON файлы
    json_files = find_json_files(config["json_directory"])
    if not json_files:
        logging.error("No JSON files found")
        return
    
    # Обрабатываем JSON файлы
    new_rows = []
    for json_file in json_files:
        row_data = process_json_file(json_file)
        if row_data:
            new_rows.append(row_data)
    
    if not new_rows:
        logging.error("No valid data extracted from JSON files")
        return
    
    logging.info(f"Processed {len(new_rows)} JSON files")
    
    # Загружаем существующие данные
    output_path = Path(config["output_excel_path"])
    existing_df = load_existing_excel(output_path)
    
    # Фильтруем новые записи
    filtered_new_rows = filter_new_records(new_rows, existing_df)
    
    if not filtered_new_rows:
        logging.info("No new records to add")
        return
    
    # Создаем DataFrame с новыми данными
    new_df = pd.DataFrame(filtered_new_rows)
    
    # Объединяем существующие и новые данные
    if not existing_df.empty:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        logging.info(f"Appending {len(new_df)} new rows to existing {len(existing_df)} rows")
    else:
        final_df = new_df
        logging.info(f"Creating new Excel file with {len(new_df)} rows")
    
    # Создаем директорию если не существует
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем в Excel
    try:
        final_df.to_excel(output_path, index=False, engine='openpyxl')
        logging.info(f"Excel file saved: {output_path}")
        logging.info(f"Total rows in file: {len(final_df)}")
        logging.info(f"New rows added: {len(filtered_new_rows)}")
    except Exception as e:
        logging.error(f"Error saving Excel file: {e}")
        raise

def validate_config(config):
    """Проверяет корректность конфигурации"""
    if not config.get("json_directory"):
        raise ValueError("json_directory not specified in config")
    
    if not config.get("output_excel_path"):
        raise ValueError("output_excel_path not specified in config")
    
    if not Path(config["json_directory"]).is_dir():
        raise FileNotFoundError(f"JSON directory does not exist: {config['json_directory']}")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    setup_logging(CONFIG.get("loglevel", "INFO"))
    
    try:
        validate_config(CONFIG)
        create_excel_report(CONFIG)
        logging.info("Script completed successfully")
    except Exception as e:
        logging.critical(f"Critical error: {e}")
        exit(1)