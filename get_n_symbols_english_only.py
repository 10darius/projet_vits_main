import sys
sys.path.append('vits_multilingual-main/text')
import symbols_multilingual_v2

# Reconstruct the symbols list for English only
_pad = ['_']
_punctuation = list(symbols_multilingual_v2._punctuation)
_latin_alphabet = list(symbols_multilingual_v2._latin_alphabet) # English uses Latin alphabet
_english_ipa_specific = symbols_multilingual_v2._english_ipa_specific # This is already a list of strings


english_symbols = (
    _pad
    + _punctuation
    + _latin_alphabet
    + _english_ipa_specific
)

# Remove duplicates and sort for a canonical list
english_symbols_set = sorted(list(set(english_symbols)))

print(len(english_symbols_set))
