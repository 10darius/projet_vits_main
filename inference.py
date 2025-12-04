from distutils.command.config import config
import time
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
def synthesize(texts,config_path,checkpoint_path,speaker=None):
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = net_g.eval()

    _ = utils.load_checkpoint(checkpoint_path, net_g, None)
    start=time.time()
    for text in texts:
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            audio = net_g.infer(x_tst, x_tst_lengths, sid=speaker,noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        wavfile.write(os.path.join("sample", "{}.wav".format(text)), hps.data.sampling_rate, audio)
    print(time.time()-start)
        
if __name__=="__main__":
    print("ATTENTION: Le checkpoint de l'époque 0 contient des poids aléatoires. L'audio généré sera du bruit.")

    # --- INFERENCE EXAMPLE: Multilingual French (speaker ID 0) ---
    texts_fr=["Bonjour le monde."]
    config_path_multi="vits_multilingual-main/configs/multilingual_fr_gh.json"
    checkpoint_path_multi="logs/multilingual_fr_gh/G_0.pth" # Latest G_0.pth
    speaker_fr = torch.LongTensor([0]) # Speaker ID 0 for French
    synthesize(texts_fr,config_path_multi,checkpoint_path_multi,speaker_fr)

    # --- INFERENCE EXAMPLE: Multilingual Ghomala (speaker ID 10) ---
    texts_gh=["bə̆gne"] # Use the Ghomala phrase provided earlier
    speaker_gh = torch.LongTensor([10]) # Speaker ID 10 for Ghomala
    synthesize(texts_gh,config_path_multi,checkpoint_path_multi,speaker_gh)

    # --- INFERENCE EXAMPLE: Monolingual Ghomala (Epoch 0) ---
    # texts=["bə̆gne"]
    # config_path="logs/ghomala_only/config.json"
    # checkpoint_path="logs/ghomala_only/G_0.pth"
    # speaker = torch.LongTensor([0]) # Speaker ID for Ghomala
    # synthesize(texts,config_path,checkpoint_path,speaker)

    # --- INFERENCE EXAMPLE: Monolingual French (Epoch 0 - if available) ---
    # texts=["Bonjour, comment allez-vous ?"]
    # config_path="logs/french_only/config.json"
    # checkpoint_path="logs/french_only/G_0.pth" # Replace with actual checkpoint if available
    # speaker = torch.LongTensor([0]) # Speaker ID for French
    # synthesize(texts,config_path,checkpoint_path,speaker)