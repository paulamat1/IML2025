import numpy as np
import soundfile as sf
from pathlib import Path
import librosa
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

    fast_file = base_path / f"{base_path.name}_fast.wav"
    if fast_file.exists():
        print(f"{fast_file.name} already exists - skip fast augmentation")
        return
    

    print(f" removing silence from {original_file.name}")
    trimmed_audio, sr = remove_silence(original_file)

    trim_len_sec = len(trimmed_audio) / sr

    if trim_len_sec < 3.0:
        print(f"{original_file.name} < 3s after trimming, skipping fast augmentation")
        return

    print(f"speeding up {original_file.name}")
    sped_audio = speed_up_with_min_length(trimmed_audio, sr, base_rate=1.25, min_length_sec=3.0)

    if sped_audio is None:
        print(f"could not find valid speed-up rate for {original_file.name}, skipping")
        return

    print(f"saving fast file {fast_file.name}")
    sf.write(str(fast_file), sped_audio, sr)


def remove_silence(input_path: Path, top_db: float = 30.0):
    audio, sr = librosa.load(str(input_path), sr=None)
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed, sr

def speed_up_with_min_length(audio: np.ndarray, sr: int, base_rate: float = 1.25,  min_length_sec: float = 3.0) -> np.ndarray | None:
    orig_len_sec = len(audio) / sr
    max_rate_to_keep_min = orig_len_sec / min_length_sec
    rate = min(base_rate, max_rate_to_keep_min)
    
    if rate <= 1.0:
        return None

    sped = pyrb.time_stretch(audio, sr, rate)

    return sped


def main():
    root_dir = Path.cwd()
    print("entering root")
    traverse_root(root_dir)


if __name__ == "__main__":
    main()