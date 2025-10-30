import librosa as lb
import os
import soundfile as sf
import numpy as np
import random as rn

TOP_DB = 30           
MIN_SEG = 1.0        
MAX_SEG = 3.0 

SEED = 1000

INPUT = "class0_not_allowed"
RAW_OUTPUT = "raw_data"
NORM_OUTPUT = "norm_data"
CLASS_LABEL = "class0" 
CLASS_DIR = "class_0"

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

    chunkNorm = peak_normalize(chunk)
    sf.write(out_path_raw, chunk, sample_rate)
    sf.write(out_path_norm, chunkNorm, sample_rate)


def pad_chunk(chunk, sample_rate):
    target_len = int(MAX_SEG * sample_rate)
    if len(chunk) < target_len:
        chunk = np.pad(chunk, (0, target_len - len(chunk)))
    return chunk


def sample_len_file(file_path):
    waveform, sample_rate = lb.load(file_path, sr=16000)

    intervals = lb.effects.split(waveform, top_db=TOP_DB)
    max_samples = int(MAX_SEG * sample_rate)
    min_samples = int(MIN_SEG * sample_rate)

    total = 0
    for start, end in intervals:
        seg = waveform[start:end]
        i = 0
        while i < len(seg):
            chunk = seg[i:i+max_samples]
            if len(chunk) < min_samples:
                break
            total += len(chunk)            
            i += max_samples
    return total


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
        total = 0
        recording_path = os.path.join(speaker_dir, recording_dir)
        for file_name in os.listdir(recording_path):
            file_path = os.path.join(recording_path, file_name)
            total += sample_len_file(file_path)
        file_len[file_name] = total
    return file_len

def assign_files(file_len):
    rn.seed(SEED)
    files = list(file_len.keys())
    rn.shuffle(files)

    total = sum(file_len.values())
    train_amount = 0.72 * total
    validation_amount = 0.18 * total

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
    train_amount = 0.8 * total
    validation_amount = 0.16 * total

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
    waveform, sample_rate = lb.load(file_path, sr=16000) #waveform is an numpy array of audio samples

    intervals = lb.effects.split(waveform, top_db=TOP_DB) #each interval contains [start, end] sample positions
    max_samples = int(MAX_SEG * sample_rate)
    min_samples = int(MIN_SEG * sample_rate)
    count = 0

    for start, end in intervals:  
        segment = waveform[start:end]
        i = 0
        while i < len(segment):
            chunk = segment[i:i+max_samples]
            if len(chunk) < min_samples:
                break
            chunk = pad_chunk(chunk, sample_rate)  
            save_chunk(chunk, sample_rate, category, recording_name, count)
            count+=1
            i+=max_samples

def test_segmentation_ratio(dir, sample_len, assign_something):
        len = sample_len(dir)
        cat = assign_something(len)

        train_time = valid_time = test_time = 0
        for something in len:
            if(cat[something] == "validation_data"):
                valid_time += len[something]
            elif(cat[something] == "test_data"):
                test_time += len[something]
            else:
                train_time += len[something]
        full_time = train_time + valid_time + test_time
        print(train_time/full_time)
        print(test_time/full_time)
        print(valid_time/full_time)

def main():    
    speaker_len = sample_len_speaker(INPUT)
    speaker_cat = assign_files(speaker_len)
    test_segmentation_ratio(INPUT,sample_len_speaker,assign_files)

    for speaker in os.listdir(INPUT):
        speaker_dir = os.path.join(INPUT,speaker)
        for recording_dir in os.listdir(speaker_dir):
            recording_name = f"{recording_dir}"
            recording_path = os.path.join(speaker_dir, recording_dir)
            for file_name in os.listdir(recording_path):
                file_path = os.path.join(recording_path, file_name)
                segment_file(file_path, speaker_cat[speaker], recording_name)

if __name__ == "__main__":
    main()
