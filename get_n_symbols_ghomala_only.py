import sys
sys.path.append('vits_multilingual-main/text')
import symbols_multilingual_v2

# Reconstruct the symbols list for Ghomala' only
_pad = ['_']
_punctuation = list(symbols_multilingual_v2._punctuation)
_latin_alphabet = list(symbols_multilingual_v2._latin_alphabet)
_ghomala_nasal_vowels = list(symbols_multilingual_v2._ghomala_nasal_vowels)
_ghomala_accented_vowels = list(symbols_multilingual_v2._ghomala_accented_vowels)
_tones = list(symbols_multilingual_v2._tones) # If the lexicon directly uses tones, include them


ghomala_symbols = (
    _pad
    + _punctuation
    + _latin_alphabet
    + _ghomala_nasal_vowels
    + _ghomala_accented_vowels
    + _tones # Include tones if they are part of the Ghomala' symbol set definition.
)

# Remove duplicates and sort for a canonical list
ghomala_symbols_set = sorted(list(set(ghomala_symbols)))

print(len(ghomala_symbols_set))
