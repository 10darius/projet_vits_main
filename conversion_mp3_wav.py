#%%
from pydub import AudioLoader, AudioWriter
import os

def convert_to_wav(mp3_path, output_dir):
    # On charge le fichier audio
    audio = AudioLoader().load(mp3_path)
    
    # Onécrit chaque track dans un fichier WAV
    for i, track in enumerate(audio.tracks):
        wav_path = os.path.join(output_dir, f"track_{i+1}.wav")
        with AudioWriter() as writer:
            writer.set_format('wav')  # Codec de sortie
            writer.set_codec('PW')     # Codec du format (pour le WAV)
            writer.set_subtype('0x00')
            #writer.
            writer.set_Channels(audio.get_n_channels())
            writer.set_Sample_Rate(audio.sample_rate)
            writer.set_BitDepth(audio.sample_width * 8)  # Passer à 16 bits pour un meilleur qualité
            
            writer.write(wav_path)
#%%
# Démarrer la conversion
mp3_path = "./dataset_bbj/clips"  # Où sont situés les fichiers .mp3 ?
output_dir = "./dataset_bbj/wav"  # Où seront les fichiers .wav sauvegardés ?

for file in os.listdir(mp3_path):
    if file.lower().endswith('.mp3'):
        full_path = os.path.join(mp3_path, file)
        output_wav_path = os.path.join(output_dir, file.replace('.mp3', '.wav'))
        convert_to_wav(full_path, output_dir)

print("Conversion terminée avec succès !")

#%%
#Avec pyaudio
import os
from pyaudio import Audio, write
import wave

def convert_mp3_to_wav_mono(mp3_path: str, output_dir: str):
    """Convertit un fichier .mp3 en .wav mono avec pyaudio."""
    try:
        # Ouvre le fichier audio MP3
        with open(mp3_path, 'rb') as f:
            audio = pyaudio.PyAudio()

            # Get audio information
            chunk_size = 1024
            stream = audio.open(format=pyaudio.paInt16,
                                channels=2 if not mono else 1,
                                rate=48000,
                                input=True,
                                output=False,
                                frames_per_buffer=chunk_size)

            # Read data in chunks
            while True:
                data = stream.read(chunk_size)
                if not data:
                    break

                # Enregistrement des données
                with wave.open(output_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16 bits
                    wf.setframerate(48000)
                    wf.writeframes(wf, data)

            stream.close()
            audio.terminate()

        print(f"Le fichier a été converti avec succès dans le répertoire {output_dir}.")
    except Exception as e:
        print(f"L'erreur suivante a été rencontrée : {e}")

#%%
# Démarrer la conversion
mp3_path = "voies.mp3"
output_dir = "resultats"

convert_mp3_to_wav_mono(mp3_path, output_dir)
#%%
# avec soundfile
import os
import soundfile as sf

def convert_mp3_to_wav_mono(mp3_path: str, output_dir: str):
    """Convertit un fichier .mp3 en .wav mono avec soundfile."""
    try:
        # Load le fichier audio MP3
        y, sr = sf.read(mp3_path)

        # Converti en mono si nécessaire
        if len(y.shape) == 1:
            pass  # Already mono
        else:
            y = y.mean(axis=1)

        # Écriture du fichier WAV mono
        sf.write(os.path.join(output_dir, os.path.splitext(mp3_path)[0] + '.wav'), y.astype('int16'), sr)

        print(f"Le fichier a été converti avec succès dans le répertoire {output_dir}.")
    except Exception as e:
        print(f"L'erreur suivante a été rencontrée : {e}")
#%%
# Démarrer la conversion
mp3_path = "voies.mp3"
output_dir = "resultats"

convert_mp3_to_wav_mono(mp3_path, output_dir)
#%%
import os
import soundfile as sf
mp3_path = "dataset_bbj/clips"  # Où sont situés les fichiers .mp3 ?
output_dir = "dataset_bbj/wav"  # Où seront les fichiers .wav sauvegardés ?
x=1
for file in os.listdir(mp3_path):
    if file.lower().endswith('.mp3'):
        try:
            with sf(file,'r',sr=sr):
                f=
            sf.info(file)
        except:
            pass
        x=x+1
        y= sf.read(file)
        print(x,"\n")
# %%
with open(mp3_path, 'rb') as f:
            audio = pyaudio.PyAudio()

            # Get audio information
            chunk_size = 1024
            stream = audio.open(format=pyaudio.paInt16,
                                channels=2 if not mono else 1,
                                #rate=48000,
                                input=True,
                                output=False,
                                frames_per_buffer=chunk_size)
#%%