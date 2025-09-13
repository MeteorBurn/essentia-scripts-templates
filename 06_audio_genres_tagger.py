#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from pathlib import Path
import logging
from mutagen import File
from mutagen.wave import WAVE
from mutagen.aiff import AIFF
from mutagen.flac import FLAC
from mutagen.mp3 import MP3

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
CONFIG = {
    "excel_file": "avantgarde_analysis.xlsx",
    "genre_source_field": "genre_discogs400",  # "genres_maest_400_v2" or "genre_discogs400"
    "file_path_field": "file_path",
    "genre_separator": "; ",
    "max_genres": 3,
    "overwrite_existing": True,
    "max_rows": None,
    "loglevel": "INFO"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def setup_logging(level: str):
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def is_wsl_environment():
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

def convert_path_for_current_env(file_path):
    if not isinstance(file_path, str):
        return str(file_path) if file_path else ""
    
    # Нормализуем Unicode
    try:
        file_path = file_path.encode('utf-8').decode('utf-8')
    except:
        pass
    
    if is_wsl_environment():
        # В WSL: всегда конвертируем Windows пути в WSL формат
        if len(file_path) >= 3 and file_path[1] == ':' and file_path[2] in ['\\', '/']:
            # Windows path: M:\path или M:/path -> /mnt/m/path
            drive = file_path[0].lower()
            rest = file_path[3:].replace('\\', '/')
            return f"/mnt/{drive}/{rest}"
        return file_path
    else:
        # В Windows: конвертируем WSL пути в Windows формат
        if file_path.startswith('/mnt/'):
            parts = file_path.split('/')
            if len(parts) > 2:
                drive = parts[2].upper()
                rest = "\\".join(parts[3:])
                return f"{drive}:\\{rest}"
        return file_path

def get_existing_genre(audio_file):
    try:
        if isinstance(audio_file, WAVE):
            if hasattr(audio_file, 'tags') and audio_file.tags:
                if 'TCON' in audio_file.tags:
                    return str(audio_file.tags['TCON'])
                ignr = audio_file.tags.get('IGNR')
                return ignr[0] if ignr else None
        elif isinstance(audio_file, MP3):
            if hasattr(audio_file, 'tags') and audio_file.tags and 'TCON' in audio_file.tags:
                return str(audio_file.tags['TCON'])
        elif isinstance(audio_file, AIFF):
            if hasattr(audio_file, 'tags') and audio_file.tags:
                if 'TCON' in audio_file.tags:
                    return str(audio_file.tags['TCON'])
                return audio_file.tags.get('GENRE', [None])[0] if 'GENRE' in audio_file.tags else None
        elif isinstance(audio_file, FLAC):
            return audio_file.get('GENRE', [None])[0] if audio_file.get('GENRE') else None
        else:
            if hasattr(audio_file, 'tags') and audio_file.tags:
                if 'TCON' in audio_file.tags:
                    return str(audio_file.tags['TCON'])
                return audio_file.get('GENRE', [None])[0] if audio_file.get('GENRE') else None
    except:
        pass
    return None

def set_genre_tag(audio_file, genre_value):
    try:
        if isinstance(audio_file, WAVE):
            if not hasattr(audio_file, 'tags') or not audio_file.tags:
                audio_file.add_tags()
            
            if hasattr(audio_file.tags, 'add'):
                from mutagen.id3 import TCON
                if 'TCON' in audio_file.tags:
                    del audio_file.tags['TCON']
                audio_file.tags.add(TCON(encoding=3, text=[genre_value]))
            else:
                audio_file.tags['IGNR'] = genre_value
                
        elif isinstance(audio_file, MP3):
            if not hasattr(audio_file, 'tags') or not audio_file.tags:
                audio_file.add_tags()
            from mutagen.id3 import TCON
            if 'TCON' in audio_file.tags:
                del audio_file.tags['TCON']
            audio_file.tags.add(TCON(encoding=3, text=[genre_value]))
            
        elif isinstance(audio_file, FLAC):
            audio_file['GENRE'] = genre_value
            
        else:
            if not hasattr(audio_file, 'tags') or not audio_file.tags:
                audio_file.add_tags()
            
            if hasattr(audio_file.tags, 'add'):
                from mutagen.id3 import TCON
                if 'TCON' in audio_file.tags:
                    del audio_file.tags['TCON']
                audio_file.tags.add(TCON(encoding=3, text=[genre_value]))
            else:
                audio_file['GENRE'] = genre_value
        return True
    except Exception as e:
        logging.error(f"Error setting genre: {e}")
        return False

def prepare_genre_string(genre_data, max_genres, separator):
    if pd.isna(genre_data) or not str(genre_data).strip():
        return None
    
    genres = [g.strip() for g in str(genre_data).split(',') if g.strip()]
    
    if max_genres and len(genres) > max_genres:
        genres = genres[:max_genres]
        logging.debug(f"Limited to {max_genres} genres: {genres}")
    
    return separator.join(genres)

def process_audio_file(file_path, genre_data, config):
    normalized_path = convert_path_for_current_env(file_path)
    
    # Дополнительная проверка пути с Path
    try:
        path_obj = Path(normalized_path)
        if not path_obj.exists():
            logging.debug(f"File not found: {normalized_path}")
            return "file_not_found"
    except Exception as e:
        logging.debug(f"Path error for {normalized_path}: {e}")
        return "path_error"
    
    try:
        audio_file = File(str(path_obj))
        if audio_file is None:
            logging.warning(f"Unsupported format: {normalized_path}")
            return "unsupported_format"
    except Exception as e:
        logging.error(f"Error loading {normalized_path}: {e}")
        return "load_error"
    
    existing_genre = get_existing_genre(audio_file)
    if existing_genre and not config["overwrite_existing"]:
        logging.debug(f"Genre exists in {path_obj.name}: {existing_genre}")
        return "skipped_existing"
    
    genre_string = prepare_genre_string(
        genre_data, 
        config["max_genres"], 
        config["genre_separator"]
    )
    
    if not genre_string:
        logging.warning(f"Empty genre data for {path_obj.name}")
        return "empty_genre"
    
    if set_genre_tag(audio_file, genre_string):
        try:
            audio_file.save()
            action = "Updated" if existing_genre else "Added"
            logging.info(f"{action} genre '{genre_string}' in {path_obj.name}")
            return "success"
        except Exception as e:
            logging.error(f"Error saving {normalized_path}: {e}")
            return "save_error"
    
    return "tag_error"

def run_genre_tagging(config):
    logging.info("Starting genre tagging process")
    
    if not Path(config["excel_file"]).is_file():
        logging.critical(f"Excel file not found: {config['excel_file']}")
        return
    
    try:
        df = pd.read_excel(config["excel_file"])
        logging.info(f"Loaded Excel file with {len(df)} rows")
    except Exception as e:
        logging.critical(f"Error reading Excel file: {e}")
        return
    
    required_columns = [config["file_path_field"], config["genre_source_field"]]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.critical(f"Missing columns: {missing_columns}")
        return
    
    if config["max_rows"]:
        df = df.head(config["max_rows"])
        logging.info(f"Processing {len(df)} rows (limited)")
    
    results = {
        "success": 0,
        "skipped_existing": 0,
        "file_not_found": 0,
        "path_error": 0,
        "unsupported_format": 0,
        "load_error": 0,
        "empty_genre": 0,
        "save_error": 0,
        "tag_error": 0
    }
    
    for index, row in df.iterrows():
        file_path = row[config["file_path_field"]]
        genre_data = row[config["genre_source_field"]]
        
        result = process_audio_file(file_path, genre_data, config)
        results[result] = results.get(result, 0) + 1
    
    logging.info("Genre tagging completed")
    logging.info(f"Successfully processed: {results['success']}")
    logging.info(f"Skipped (existing genre): {results['skipped_existing']}")
    logging.info(f"Errors: {sum(results.values()) - results['success'] - results['skipped_existing']}")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    setup_logging(CONFIG.get("loglevel", "INFO"))
    
    try:
        run_genre_tagging(CONFIG)
    except Exception as e:
        logging.critical(f"Critical error: {e}")
        exit(1)