# preprocess_monolingual_data.py

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
PROCESSED_FILES = {
    'train': {
        'fr': "filelists/train_fr.txt",
        'gh': "filelists/train_gh.txt",
        'en': "filelists/train_en.txt",
    },
    'val': {
        'fr': "filelists/val_fr.txt",
        'gh': "filelists/val_gh.txt",
        'en': "filelists/val_en.txt",
    }
}


# Path to the lexicon for Ghomala`
LEXICON_PATH = "vits_multilingual-main/ghomala_lexicon.tsv"

# ---

def process_data_monolingual():
    """
    Reads raw filelists for each language, processes them, and writes
    separate training and validation filelists for each language.
    """
    os.makedirs("filelists", exist_ok=True)
    
    for split in ['train', 'val']:
        print(f"--- Processing {split} split for monolingual files ---")
        
        for lang_key, output_path in PROCESSED_FILES[split].items():
            all_lines_processed_for_lang = []
            
            # Find the source config for this language
            source = next(item for item in DATA_SOURCES[split] if item["lang"] == lang_key)
            
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
                        audio_filename, text = line.strip().split("|", 1)
                        
                        full_audio_path = os.path.join(audio_prefix, audio_filename).replace("\\", "/")
                        
                        try:
                            _ = os.path.getsize(full_audio_path) 
                        except OSError:
                            print(f"WARNING: Audio file not found or inaccessible: '{full_audio_path}'. Skipping line: '{line.strip()}'")
                            continue
                            
                        speaker_id = speaker_id_start # For monolingual, we still use the base ID for consistency. 
                        
                        sequence = multilingual_text_to_sequence_v2(text, lang, LEXICON_PATH)
                        
                        if not sequence:
                            print(f"WARNING: No sequence generated for text: '{text}'. Skipping.")
                            continue
                            
                        str_sequence = " ".join(map(str, sequence))
                        all_lines_processed_for_lang.append(f"{full_audio_path}|{speaker_id}|{str_sequence}")

                    except Exception as e:
                        print(f"ERROR processing line: {line.strip()}")
                        print(f"Reason: {e}")
            
            random.shuffle(all_lines_processed_for_lang)
            
            print(f"Writing {len(all_lines_processed_for_lang)} lines to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f_out:
                for line in all_lines_processed_for_lang:
                    f_out.write(line + "\n")


def main():
    """Main entry point for monolingual preprocessing."""
    print("--- Starting Monolingual Data Preprocessing ---")
    
    process_data_monolingual()
    
    print("\n--- Monolingual Preprocessing Complete ---")
    print("Your training-ready filelists are:")
    for split_key, lang_paths in PROCESSED_FILES.items():
        for lang_key, path in lang_paths.items():
            print(f"- {path} ({split_key} {lang_key})")
    print("\nNext Steps:")
    print("1. Ensure 'ghomala_lexicon.tsv' is populated for best Ghomala' quality.")
    print("2. Use the generated filelists with your monolingual configuration JSONs.")
    print("3. Start training!")

if __name__ == "__main__":
    main()
