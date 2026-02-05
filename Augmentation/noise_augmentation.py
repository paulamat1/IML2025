import pandas as pd
import numpy as np
import soundfile as sf
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
import librosa






def traverse_root(base_path:Path):

    class_dirs = [
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.lower().startswith("class")
    ]

    for class_dir in class_dirs:
        print(f"processing directory: {class_dir}")
        for person_folder in class_dir.iterdir():
            if not person_folder.is_dir():
                continue
            else:
                for recording_folder in person_folder.iterdir():
                    if not recording_folder.is_dir():
                        continue
                    else:
                        process_recording_folder(recording_folder)


def process_recording_folder(base_path: Path):
    wav_files = [
        f for f in base_path.glob("*.wav")
        if not f.name.endswith("_noise.wav")
        if not f.name.endswith("_chop.wav")
        if not f.name.endswith("_fast.wav")
        if not f.name.endswith("_slow.wav")
    ]

    if not wav_files:
        print(f"empty folder: {base_path.name}")
        return
    
    original_file = wav_files[0] # in general i assume there is only one og video
    og_path = base_path / f"{base_path.name}.wav"

    if original_file != og_path:
        if og_path.exists():
            print(f"{og_path} already exists - skip rename of {original_file.name}")
        else:
            print(f"renaming {original_file.name} to {og_path.name}")
            original_file.rename(og_path)
            original_file=og_path
    
    noise_file = base_path / f"{base_path.name}_noise.wav"

    if noise_file.exists():
        print(f"{noise_file.name} already exists - skip augmentation")
        return

    print(f"removing silence from {original_file.name}")
    trimmed_file,sr = remove_silence(original_file)

    print(f"adding noise to {original_file.name}")
    noise_audio= add_noise(trimmed_file,sr)

    print(f"saving augmented file {noise_file.name}")
    sf.write(str(noise_file),noise_audio,sr)


def remove_silence(input_path: Path, top_db:float =30.0):
    audio,sr=librosa.load(str(input_path),sr=None)
    trimmed,_=librosa.effects.trim(audio,top_db=top_db)
    return trimmed,sr

def add_noise(audio:np.ndarray,sr:int,snr_min: float = 10, snr_max: float = 30): # the noise factor can be played with!!
    snr_db = np.random.uniform(snr_min, snr_max)
    sig_power = np.mean(audio ** 2)
    if sig_power==0:
        return audio
    
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    noise_audio = audio + noise
    noise_audio = np.clip(noise_audio, -1.0, 1.0)
    return noise_audio


def main():
    root_dir = Path.cwd()
    traverse_root(root_dir)



if __name__ == "__main__":
    main()