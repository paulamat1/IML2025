import numpy as np
import soundfile as sf
from pathlib import Path
import librosa


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

    chop_file = base_path / f"{base_path.name}_chop.wav"
    if chop_file.exists():
        print(f"{chop_file.name} already exists - skip chop augmentation")
        return
    print(f"removing silence from {original_file.name}")
    trimmed_audio, sr = remove_silence(original_file)

    print(f"chop and shuffle on {original_file.name}")
    chopped_audio = random_chop_and_shuffle(trimmed_audio, num_segments=50)

    print(f"saving chopped file {chop_file.name}")
    sf.write(str(chop_file), chopped_audio, sr)

    


def remove_silence(input_path: Path, top_db:float =30.0):
    audio,sr=librosa.load(str(input_path),sr=None)
    trimmed,_=librosa.effects.trim(audio,top_db=top_db)
    return trimmed,sr


def random_chop_and_shuffle(audio: np.ndarray, num_segments: int = 50) -> np.ndarray:
    segments = np.array_split(audio, num_segments)
    segments = [seg for seg in segments if seg.size > 0]
    if len(segments) == 0:
        return audio
    
    perm = np.random.permutation(len(segments))
    shuffled_segments = [segments[i] for i in perm]
    return np.concatenate(shuffled_segments)


def main():
    root_dir = Path.cwd()
    print("entering root")
    traverse_root(root_dir)



if __name__ == "__main__":
    main()