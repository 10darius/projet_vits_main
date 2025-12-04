
import numpy as np
from scipy.io.wavfile import read
import sys

def inspect_audio(file_path):
    try:
        sampling_rate, data = read(file_path)
        print(f"Successfully read {file_path}")
        print(f"  Sampling Rate: {sampling_rate}")
        print(f"  Data Shape: {data.shape}")
        print(f"  Data Type: {data.dtype}")
        
        # If stereo, convert to mono for analysis
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        max_val = np.max(np.abs(data))
        mean_val = np.mean(data)
        
        print(f"  Max Absolute Value: {max_val}")
        print(f"  Mean Value: {mean_val}")
        
        if max_val == 0:
            print("  [CONCLUSION] The audio file is completely silent (all zeros).")
        else:
            print("  [CONCLUSION] The audio file contains non-zero data.")
            
    except Exception as e:
        print(f"Error reading or analyzing {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_to_inspect = sys.argv[1]
        inspect_audio(file_to_inspect)
    else:
        print("Please provide the path to a WAV file.")
