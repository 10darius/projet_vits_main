# preprocess_multilingual_data_v2.py

import os
import sys
import random
from tqdm import tqdm

# Add the project directory to the path to allow imports from `text`
sys.path.append('vits_multilingual-main')
from text.cleaners_multilingual_v2 import multilingual_text_to_sequence_v2

# --- Configuration ---

# This script now reads directly from the dataset-specific filelists
# and combines them.

# --- Paths and Speaker ID Mappings ---
DATA_SOURCES = {
    'train': [
        {
            'lang': 'fr',
            'speaker_id_start': 0,
            'filelist': 'vits_multilingual-main/dataset_fr/train_fr.txt',
            'audio_prefix': '' # French filelist uses absolute paths
        },
        {
            'lang': 'gh',
            'speaker_id_start': 10,
            'filelist': 'vits_multilingual-main/dataset_bbj/train.txt',
            'audio_prefix': 'vits_multilingual-main/' # Ghomala' filelist audio_filenames start with 'dataset_bbj/wav/'
        },
        {
            'lang': 'en',
            'speaker_id_start': 20,
            'filelist': 'vits_multilingual-main/dataset/me_train.txt',
            'audio_prefix': 'vits_multilingual-main/dataset/' # English filelist audio_filenames start with 'wavs/'
        }
    ],
    'val': [
        {
            'lang': 'fr',
            'speaker_id_start': 0,
            'filelist': 'vits_multilingual-main/dataset_fr/val_fr.txt',
            'audio_prefix': '' # French filelist uses absolute paths
        },
        {
            'lang': 'gh',
            'speaker_id_start': 10,
            'filelist': 'vits_multilingual-main/dataset_bbj/test.txt', # Using test as validation
            'audio_prefix': 'vits_multilingual-main/' # Ghomala' filelist audio_filenames start with 'dataset_bbj/wav/'
        },
        {
            'lang': 'en',
            'speaker_id_start': 20,
            'filelist': 'vits_multilingual-main/dataset/me_val.txt',
            'audio_prefix': 'vits_multilingual-main/dataset/' # English filelist audio_filenames start with 'wavs/'
        }
    ]
}

# The output files will be in the format required by the VITS data loader:
# path/to/wav.wav|speaker_id|id1 id2 id3...
PROCESSED_TRAIN_FILE = "filelists/train_processed.txt"
PROCESSED_VALIDATION_FILE = "filelists/val_processed.txt"

# Path to the lexicon for Ghomala'
LEXICON_PATH = "vits_multilingual-main/ghomala_lexicon.tsv"

# ---

def process_data():
    """
    Reads all raw filelists, processes them, and writes final training files.
    """
    os.makedirs("filelists", exist_ok=True)
    
    for split in ['train', 'val']:
        print(f"--- Processing {split} split ---")
        
        all_lines_processed = []
        
        for source in DATA_SOURCES[split]:
            lang = source['lang']
            filelist_path = source['filelist']
            speaker_id_start = source['speaker_id_start']
            audio_prefix = source['audio_prefix']
            
            if not os.path.exists(filelist_path):
                print(f"WARNING: Filelist not found, skipping: {filelist_path}")
                continue

            print(f"Reading {filelist_path} for language '{lang}'...")
            with open(filelist_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f"Processing {os.path.basename(filelist_path)}"):
                    try:
                        # Format: audio_filename.wav|text
                        audio_filename, text = line.strip().split("|", 1)
                        
                        full_audio_path = os.path.join(audio_prefix, audio_filename).replace("\\", "/")
                        
                        try:
                            # Use os.path.getsize to check for existence and accessibility,
                            # as os.path.exists can sometimes return True for inaccessible files on Windows.
                            # Also, this aligns with the eventual check in data_utils.py that failed.
                            _ = os.path.getsize(full_audio_path) 
                        except OSError:
                            print(f"WARNING: Audio file not found or inaccessible: '{full_audio_path}'. Skipping line: '{line.strip()}'")
                            continue
                            
                        # For simplicity, we assign a single speaker ID per language group for now.
                        # This can be expanded if speaker information is available in the filelists.
                        speaker_id = speaker_id_start
                        
                        # Convert text to sequence of IDs using the V2 cleaner
                        sequence = multilingual_text_to_sequence_v2(text, lang, LEXICON_PATH)
                        
                        if not sequence:
                            print(f"WARNING: No sequence generated for text: '{text}'. Skipping.")
                            continue
                            
                        str_sequence = " ".join(map(str, sequence))
                        all_lines_processed.append(f"{full_audio_path}|{speaker_id}|{str_sequence}")

                    except Exception as e:
                        print(f"ERROR processing line: {line.strip()}")
                        print(f"Reason: {e}")
        
        # Shuffle the combined list before writing
        random.shuffle(all_lines_processed)
        
        output_path = PROCESSED_TRAIN_FILE if split == 'train' else PROCESSED_VALIDATION_FILE
        print(f"Writing {len(all_lines_processed)} lines to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line in all_lines_processed:
                f_out.write(line + "\n")

def main():
    """Main entry point."""
    print("--- Starting Multilingual Data Preprocessing (V2) ---")
    
    process_data()
    
    print("\n--- Preprocessing Complete ---")
    print("Your training-ready filelists are:")
    print(f"- {PROCESSED_TRAIN_FILE}")
    print(f"- {PROCESSED_VALIDATION_FILE}")
    print("\nNext Steps:")
    print("1. Ensure 'ghomala_lexicon.tsv' is populated for best Ghomala' quality.")
    print("2. Update your model's configuration JSON to use these 'processed' filelists.")
    print("3. Start training!")

if __name__ == "__main__":
    main()
