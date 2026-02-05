import numpy as np
import soundfile as sf
from pathlib import Path
import librosa # pip install librosa - if needed
import pyrubberband as pyrb


def traverse_root(base_path: Path):
    class_dirs = [
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.lower().startswith("class")
    ]

    for class_dir in class_dirs:
        print(f"processing directory: {class_dir}")
        for person_folder in class_dir.iterdir():
            if not person_folder.is_dir():
                continue
            for recording_folder in person_folder.iterdir():
                if not recording_folder.is_dir():
                    continue
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
        print(f"empty folder (no base wav): {base_path.name}")
        return
    
    original_file = wav_files[0]
    og_path = base_path / f"{base_path.name}.wav"

    if original_file != og_path:
        if og_path.exists():
            print(f"{og_path} already exists - skip rename of {original_file.name}")
            original_file = og_path
        else:
            print(f"renaming {original_file.name} to {og_path.name}")
            original_file.rename(og_path)
            original_file = og_path

    slow_file = base_path / f"{base_path.name}_slow.wav"
    if slow_file.exists():
        print(f"{slow_file.name} already exists - skip slow augmentation")
        return
    
    print(f"removing silence from {original_file.name}")
    trimmed_audio, sr = remove_silence(original_file)

    if trimmed_audio.size == 0:
        print(f"trimmed audio is empty for {original_file.name}, skipping")
        return
    
    print(f"slowing down {original_file.name}")
    slowed_audio = slow_down(trimmed_audio, sr,rate=0.8)

    print(f"saving slowed file {slow_file.name}")
    sf.write(str(slow_file), slowed_audio, sr) 

def remove_silence(input_path: Path, top_db: float = 30.0):
    audio, sr = librosa.load(str(input_path), sr=None)
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed, sr

def slow_down(audio: np.ndarray, sr:int, rate: float = 0.8) -> np.ndarray:
    slowed = pyrb.time_stretch(audio, sr, rate)
    return slowed
                    

def main():
    root_dir = Path.cwd()
    print("entering root")
    traverse_root(root_dir)


if __name__ == "__main__":
    main()