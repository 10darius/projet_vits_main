# create_lexicon_template.py

import re
from collections import Counter
import glob

# --- Configuration ---
# Path to the text files of the Ghomala' dataset
# Assumes a structure like dataset_gh/transcript.txt or similar
DATASET_PATH_PATTERN = "dataset_gh/*.txt" 
OUTPUT_FILE = "ghomala_lexicon_template.tsv"
# ---

# Regex to find words
_word_re = re.compile(r"[a-zA-ZÀ-ÿʉəɛɔŋ'-]+")

def create_template():
    """
    Scans the dataset text files, extracts all unique words,
    and creates a frequency-sorted template for the lexicon.
    """
    word_counts = Counter() 
    
    print(f"Scanning files matching pattern: {DATASET_PATH_PATTERN}")
    
    files = glob.glob(DATASET_PATH_PATTERN)
    if not files:
        print(f"ERROR: No files found for pattern '{DATASET_PATH_PATTERN}'.")
        print("Please check the DATASET_PATH_PATTERN variable in this script.")
        # Create a dummy file to show the format
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("# Ghomala' Lexicon Template\n")
            f.write("# Format: word<TAB>p h o n e m e s\n")
            f.write("# Replace the right side with IPA transcription.\n")
            f.write("# -------------------------------------------\
")
            f.write("bonjour\tb ɔ̃ ʒ u ʁ\n")
            f.write("ghomala'\tɡ o m a l a ¹\n")
        return

    for filepath in files:
        print(f"Processing {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                # Assuming format like: path/to/wav|TEXT
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    text = parts[-1]
                    words = _word_re.findall(text.lower())
                    word_counts.update(words)

    print(f"Found {len(word_counts)} unique words.")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Ghomala' Lexicon Template\n")
        f.write("# Format: word<TAB>p h o n e m e s\n")
        f.write("# Replace the right side with IPA transcription, including tones.\n")
        f.write("# Words are sorted by frequency (most common first).\n")
        f.write("# ---------------------------------------------------\
")
        
        for word, count in word_counts.most_common():
            # Write the word and a placeholder for the user to fill in
            f.write(f"{word}\t# TODO ({count} occurrences)\n")
            
    print(f"Successfully created lexicon template at '{OUTPUT_FILE}'.")
    print("\nNEXT STEP: Edit this file to add the correct IPA pronunciations.")


if __name__ == "__main__":
    create_template()
