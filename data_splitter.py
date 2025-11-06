import librosa as lb
import os
import soundfile as sf
import numpy as np
import random as rn

TOP_DB = 35
INTERNAL_DB = 25

MAX_SEG = 3.0       
FRAME_LENGTH = 2048       
HOP_LENGTH = 256   

SEED = 200

INPUT = "class1_allowed"
RAW_OUTPUT = "raw_data"
NORM_OUTPUT = "norm_data"
CLASS_LABEL = "class1" 
CLASS_DIR = "class_1"

def peak_normalize(chunk):
    peak = np.max(np.abs(chunk))
    if peak != 0:
        return chunk / peak
    else:
        return chunk
   
def save_chunk(chunk, sample_rate, category, recording_name, count):
    raw_dir = os.path.join(RAW_OUTPUT, category, CLASS_DIR, recording_name)
    norm_dir = os.path.join(NORM_OUTPUT, category, CLASS_DIR, recording_name)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)

    base_name = f"{CLASS_LABEL}_{recording_name}_{count:03d}.wav"
    out_path_raw = os.path.join(raw_dir,base_name)
    out_path_norm = os.path.join(norm_dir, base_name)

    chunk = np.asarray(chunk, dtype=np.float32)
    chunkNorm = peak_normalize(chunk)
    sf.write(out_path_raw, chunk, sample_rate, subtype="PCM_16")
    sf.write(out_path_norm, chunkNorm, sample_rate, subtype="PCM_16")


def pad_chunk(chunk, sample_rate):
    target_len = int(MAX_SEG * sample_rate)
    if len(chunk) < target_len:
        chunk = np.pad(chunk, (0, target_len - len(chunk)))
    return chunk


def sample_len_file(file_path):
    waveform, _ = lb.load(file_path, sr=16000, mono=True) #waveform is an numpy array of audio samples
    waveform_trim, _ = lb.effects.trim(waveform, top_db=TOP_DB, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    internal_segments = lb.effects.split(waveform_trim, top_db=INTERNAL_DB, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    
    if len(internal_segments) == 0:
        return len(waveform_trim)
    else:
        chunks = []
        for start, end in internal_segments:
            chunks.append(waveform_trim[start:end])
        waveform_clean = np.concatenate(chunks)

    return len(waveform_clean)

def sample_len_speaker(speakers_path):
    speaker_len = {}
    for speaker in os.listdir(speakers_path):
        speaker_dir = os.path.join(speakers_path, speaker)
        total = 0
        for recording_dir in os.listdir(speaker_dir):
            recording_path = os.path.join(speaker_dir, recording_dir)
            for file_name in os.listdir(recording_path):
                file_path = os.path.join(recording_path, file_name)
                total += sample_len_file(file_path)
        speaker_len[speaker] = total
    return speaker_len

def sample_len_per_file(speaker_dir):
    file_len = {}
    for recording_dir in os.listdir(speaker_dir):
        recording_path = os.path.join(speaker_dir, recording_dir)
        for file_name in os.listdir(recording_path):
            file_path = os.path.join(recording_path, file_name)
            file_len[file_name] = sample_len_file(file_path)
    return file_len

def assign_files(file_len):
    rn.seed(SEED)
    files = list(file_len.keys())
    rn.shuffle(files)

    total = sum(file_len.values())
    train_amount = 0.75 * total
    validation_amount = 0.15 * total

    train_sum = val_sum = 0

    file_cat = {}
    for file in files:
        dur = file_len[file]
        if train_sum + dur <= train_amount:
            file_cat[file] = "train_data"
            train_sum += dur
        elif val_sum + dur <= validation_amount:
            file_cat[file] = "validation_data"
            val_sum += dur
        else:
            file_cat[file] = "test_data"
    return file_cat


def assign_speakers(speaker_len):
    rn.seed(SEED)
    speakers = list(speaker_len.keys())
    rn.shuffle(speakers)

    total = sum(speaker_len.values())
    train_amount = 0.7 * total
    validation_amount = 0.2 * total

    train_sum = val_sum = 0

    speaker_cat = {}
    for speaker in speakers:
        dur = speaker_len[speaker]
        if train_sum + dur <= train_amount:
            speaker_cat[speaker] = "train_data"
            train_sum += dur
        elif val_sum + dur <= validation_amount:
            speaker_cat[speaker] = "validation_data"
            val_sum += dur
        else:
            speaker_cat[speaker] = "test_data"
    return speaker_cat


def segment_file(file_path, category, recording_name):
    waveform, sample_rate = lb.load(file_path, sr=16000, mono=True) #waveform is an numpy array of audio samples

    waveform_trim, _ = lb.effects.trim(waveform, top_db=TOP_DB, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    internal_segments = lb.effects.split(waveform_trim, top_db=INTERNAL_DB, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

    if len(internal_segments) == 0:
        waveform_clean = waveform_trim
    else:
        chunks = []
        for start, end in internal_segments:
            chunks.append(waveform_trim[start:end])
        waveform_clean = np.concatenate(chunks)

    max_samples = int(MAX_SEG * sample_rate)
    count = 0
    i = 0
    while i < len(waveform_clean):
        chunk = waveform_clean[i:i+max_samples]
        if len(chunk) < max_samples:
            chunk = pad_chunk(chunk, sample_rate)   
        save_chunk(chunk, sample_rate, category, recording_name, count)
        count+=1
        i+=max_samples

def test_segmentation_ratio(dir, sample_len, assign_something):
        len_dir = sample_len(dir)
        cat = assign_something(len_dir)

        train_time = valid_time = test_time = 0
        for something in len_dir:
            if(cat[something] == "validation_data"):
                valid_time += len_dir[something]
            elif(cat[something] == "test_data"):
                test_time += len_dir[something]
            elif(cat[something] == "train_data"):
                train_time += len_dir[something]
            else:
                raise ValueError(f"Unknown category: {len_dir[something]}")
        full_time = train_time + valid_time + test_time
        print(train_time/full_time)
        print(test_time/full_time)
        print(valid_time/full_time)

def main():    
    """speaker_len = sample_len_speaker(INPUT)
    speaker_cat = assign_speakers(speaker_len)
    test_segmentation_ratio(INPUT,sample_len_speaker,assign_speakers)"""

    for speaker in os.listdir(INPUT):
        speaker_dir = os.path.join(INPUT,speaker)
        
        file_len = sample_len_per_file(speaker_dir)
        file_cat = assign_files(file_len)
        test_segmentation_ratio(speaker_dir, sample_len_per_file, assign_files)

        for recording_dir in os.listdir(speaker_dir):
            recording_name = f"{recording_dir}"
            recording_path = os.path.join(speaker_dir, recording_dir)
            for file_name in os.listdir(recording_path):
                file_path = os.path.join(recording_path, file_name)
                segment_file(file_path, file_cat[file_name], recording_name)

if __name__ == "__main__":
    main()
