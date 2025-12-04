# preprocess_multilingual_data_v1.py

import os
from tqdm import tqdm
from text.cleaners_multilingual import multilingual_text_to_sequence

# --- Configuration ---

# This input file should contain your raw data in the format:
# path/to/wav.wav|speaker_id|lang_code|Raw text here
#
# Example:
# D:/data/fr/wav1.wav|0|fr|Ceci est un test.
# D:/data/gh/wav1.wav|1|gh|Texte en Ghomala'.
RAW_TRAIN_FILE = "filelists/train_raw.txt"
RAW_VALIDATION_FILE = "filelists/val_raw.txt"

# The output files will be in the format required by the VITS data loader:
# path/to/wav.wav|speaker_id|id1 id2 id3...
PROCESSED_TRAIN_FILE = "filelists/train_processed.txt"
PROCESSED_VALIDATION_FILE = "filelists/val_processed.txt"

# Path to the lexicon for Ghomala'
LEXICON_PATH = "ghomala_lexicon.tsv"

# ---

def process_filelist(input_path, output_path):
    """
    Reads a raw filelist, converts text to a sequence of symbol IDs,
    and writes the processed filelist.
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Raw filelist not found at '{input_path}'.")
        print("Please create it first with the format: path|speaker_id|lang|text")
        # Create a dummy file for demonstration
        with open(input_path, "w", encoding="utf-8") as f:
            f.write("dataset_fr/example.wav|0|fr|Ceci est un exemple.\n")
            f.write("dataset_gh/example.wav|1|gh|Ceci est un exemple ghomala.\n")
        print(f"A dummy file has been created at '{input_path}'. Please edit it.")
        return

    print(f"Processing {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        lines = f_in.readlines()
        for line in tqdm(lines, desc=f"Preprocessing {os.path.basename(input_path)}"):
            parts = line.strip().split("|")
            if len(parts) != 4:
                print(f"WARNING: Skipping malformed line: {line.strip()}")
                continue
            
            path, speaker_id, lang, text = parts
            
            try:
                # Convert text to a sequence of IDs
                sequence = multilingual_text_to_sequence(text, lang, LEXICON_PATH)
                
                # Write the processed line to the output file
                str_sequence = " ".join(map(str, sequence))
                f_out.write(f"{path}|{speaker_id}|{str_sequence}\n")
            
            except Exception as e:
                print(f"ERROR processing line: {line.strip()}")
                print(f"Reason: {e}")

    print(f"Successfully created processed file at '{output_path}'.")


def main():
    """Main entry point."""
    print("--- Starting Multilingual Data Preprocessing ---")
    
    # Ensure the output directory exists
    os.makedirs("filelists", exist_ok=True)
    
    # Process both training and validation files
    process_filelist(RAW_TRAIN_FILE, PROCESSED_TRAIN_FILE)
    process_filelist(RAW_VALIDATION_FILE, PROCESSED_VALIDATION_FILE)
    
    print("\n--- Preprocessing Complete ---")
    print(f"Your training-ready filelists are:")
    print(f"- {PROCESSED_TRAIN_FILE}")
    print(f"- {PROCESSED_VALIDATION_FILE}")
    print("\nYou can now use these files in your model's configuration JSON.")

if __name__ == "__main__":
    main()
