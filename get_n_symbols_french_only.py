import sys
sys.path.append('vits_multilingual-main/text')
import symbols_multilingual_v2

# Reconstruct the symbols list for French only
# Based on how symbols_multilingual_v2.py assembles the full symbols list
_pad = ['_']
_punctuation = list(symbols_multilingual_v2._punctuation) # Assuming all punctuation is relevant to French
_latin_alphabet = list(symbols_multilingual_v2._latin_alphabet) # Assuming all latin alphabet is relevant to French
_vowels = list(symbols_multilingual_v2._vowels) # Assuming all IPA vowels are relevant to French
_semivowels = list(symbols_multilingual_v2._semivowels) # Assuming all IPA semivowels are relevant to French
_consonants = list(symbols_multilingual_v2._consonants) # Assuming all IPA consonants are relevant to French
_french_specific = list(symbols_multilingual_v2._french_specific)


french_symbols = (
    _pad
    + _punctuation
    + _latin_alphabet
    + _vowels
    + _semivowels
    + _consonants
    + _french_specific
)

# Remove duplicates and sort for a canonical list
french_symbols_set = sorted(list(set(french_symbols)))

print(len(french_symbols_set))
