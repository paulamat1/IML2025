from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union, TYPE_CHECKING

import numpy as np
import soundfile as sf
import librosa

if TYPE_CHECKING:
    import torch 
else:
    try:
        import torch  # runtime import
    except ImportError:
        torch = None  # runtime fallback

# config
@dataclass(frozen=True)
class PreprocessConfig:
    # audio
    sr: int = 16000

    # silence removal
    top_db: int = 35
    internal_db: int = 25
    frame_length: int = 2048
    hop_length_trim: int = 256

    # chunking
    chunk_sec: float = 3.0
    pad_last_chunk: bool = True 

    # spectrogram 
    n_fft: int = 400 
    hop: int = 160 
    n_mels: int = 80
    fmin: int = 20
    fmax: int = 7600

    # fixed width target
    target_frames: int = 300 




# audio loading

def load_audio_mono_resample(
    wav_path: str,
    cfg: PreprocessConfig,
) -> np.ndarray:
    """
    Loads audio from disk, converts to mono, and resamples to cfg.sr (16 khz).

    - soundfile for reading 
    - averages channels to mono.
    - resamples with librosa if SR differs.
    """
    y, sr = sf.read(wav_path, always_2d=False)

    # to mono
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.mean(axis=1)

    
    y = np.asarray(y, dtype=np.float32)

    # resample if needed
    if sr != cfg.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr).astype(np.float32)

    return y


# silence trimming and internal silence removal
def remove_silence(
    y: np.ndarray,
    cfg: PreprocessConfig,
) -> np.ndarray:
    """
    Removes:
    - leading/trailing silence (librosa.effects.trim)
    - internal silence regions (librosa.effects.split + concatenate)
    """
    if y.size == 0:
        return y

    y_trim, _ = librosa.effects.trim(
        y,
        top_db=cfg.top_db,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length_trim,
    )

    intervals = librosa.effects.split(
        y_trim,
        top_db=cfg.internal_db,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length_trim,
    )

    if len(intervals) == 0:
        return y_trim

    parts = [y_trim[start:end] for start, end in intervals]
    y_clean = np.concatenate(parts) if parts else y_trim
    return y_clean


# chunking into 3 sec segments
def chunk_waveform(
    y: np.ndarray,
    cfg: PreprocessConfig,
) -> List[np.ndarray]:
    """
    Splits cleaned waveform into fixed-length chunks of cfg.chunk_sec

    """
    chunk_len = int(round(cfg.chunk_sec * cfg.sr))
    if y.size == 0:
        return []

    chunks: List[np.ndarray] = []
    i = 0
    while i < len(y):
        c = y[i : i + chunk_len]
        if len(c) < chunk_len:
            if not cfg.pad_last_chunk:
                break
            c = np.pad(c, (0, chunk_len - len(c)), mode="constant")
        chunks.append(c.astype(np.float32))
        i += chunk_len

    return chunks



# generating the spectogram

def compute_logmel_cmvn(
    y: np.ndarray,
    cfg: PreprocessConfig,
) -> np.ndarray:
    """
    Computes log-Mel spectrogram and applies CMVN normalization per chunk,

    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )

    S_db = librosa.power_to_db(S, ref=np.max)
    mu = float(S_db.mean())
    std = float(S_db.std()) + 1e-10
    M = (S_db - mu) / std
    return M.astype(np.float32)


def fix_width_center(
    M: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    """
    Ensures spectrogram width is exactly target_frames
    """
    n_mels, T = M.shape
    if T == target_frames:
        return M
    if T > target_frames:
        s = (T - target_frames) // 2
        return M[:, s : s + target_frames]
    return np.pad(M, ((0, 0), (0, target_frames - T)), mode="constant")



# full pipeline for the model
def preprocess_recording_to_batch(
    wav_path: str,
    cfg: Optional[PreprocessConfig] = None,
    return_torch: bool = True,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    wav_path -> cleaned waveform -> 3s chunks -> logmel -> fixed width -> batch

   
    """
    cfg = cfg or PreprocessConfig()

    y = load_audio_mono_resample(wav_path, cfg)
    y = remove_silence(y, cfg)
    chunks = chunk_waveform(y, cfg)

    if len(chunks) == 0:
        # return an empty batch consistently
        empty = np.zeros((0, 1, cfg.n_mels, cfg.target_frames), dtype=np.float32)
        if return_torch and torch is not None:
            return torch.from_numpy(empty)
        return empty

    specs: List[np.ndarray] = []
    for c in chunks:
        M = compute_logmel_cmvn(c, cfg)
        M = fix_width_center(M, cfg.target_frames)  # [80, 300]
        specs.append(M)

    batch = np.stack(specs, axis=0)                 # [N, 80, 300]
    batch = batch[:, None, :, :]                    # [N, 1, 80, 300]

    if return_torch and torch is not None:
        return torch.from_numpy(batch.astype(np.float32))
    return batch.astype(np.float32)