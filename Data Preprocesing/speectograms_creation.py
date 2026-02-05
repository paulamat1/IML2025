import argparse, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

SR = 16000            # docelowy sample rate
N_FFT = 400           # ~25 ms przy 16 kHz
HOP = 160             # ~10 ms
N_MELS = 80
FMIN, FMAX = 20, 7600
SEG_SEC = 3.0
TARGET_FR = int(round(SEG_SEC * SR / HOP))  # ≈300 ramek

def compute_logmel(y, sr=SR):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)     # 0 dB = max
    mu, std = S_db.mean(), S_db.std() + 1e-10    # CMVN per-utterance
    return (S_db - mu) / std                     # [n_mels, T]

def center_trim_or_pad(M, target_frames=TARGET_FR):
    n_mels, T = M.shape
    if T == target_frames: return M
    if T > target_frames:
        s = (T - target_frames) // 2
        return M[:, s:s+target_frames]
    return np.pad(M, ((0,0),(0, target_frames-T)), mode="constant")

def save_png(matrix, path_png):
    plt.figure(figsize=(3, 3), dpi=150)
    librosa.display.specshow(matrix, sr=SR, hop_length=HOP)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path_png, bbox_inches="tight", pad_inches=0)
    plt.close()

def process_one(wav_path: Path, in_root: Path, out_root: Path, resample: bool):
    rel = wav_path.relative_to(in_root)
    out_base = (out_root / rel).with_suffix("")      # bez rozszerzenia
    out_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        y, sr = sf.read(str(wav_path), always_2d=False)
        if y.ndim > 1:                               # do mono
            y = y.mean(axis=1)
        if resample and sr != SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
            sr = SR
        elif sr != SR and not resample:
            raise ValueError(f"SR={sr} but resampling disabled")

        if len(y) < int(0.5 * SR):                   # skrajnie krótkie (nie powinno się zdarzyć)
            return "short", str(rel)

        M = compute_logmel(y, sr=sr).astype(np.float32)
        M = center_trim_or_pad(M, TARGET_FR)

        np.save(out_base.with_suffix(".npy"), M)
        #save_png(M, out_base.with_suffix(".png"))
        return "ok", str(rel)
    except Exception as e:
        return "err", f"{rel} :: {type(e).__name__}: {e}"

def run_tree(src_root: Path, dst_root: Path, workers: int, resample: bool):
    files = list(src_root.rglob("*.wav"))
    if not files:
        print(f"[warn] Brak plików .wav w: {src_root}")
        return {"ok":0,"err":0,"short":0}
    stats = {"ok":0,"err":0,"short":0}
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        futs = [ex.submit(process_one, p, src_root, dst_root, resample) for p in files]
        for f in as_completed(futs):
            status, msg = f.result()
            stats[status] += 1
            if status != "ok" and stats[status] <= 10:
                print(f"[{status.upper()}] {msg}")
    return stats

def main():
    ap = argparse.ArgumentParser(description="Generate log-Mel (.npy + .png) for 3s .wav files, preserving folders.")
    ap.add_argument("--in-root", required=True, help="Katalog wejściowy (z train_data/… itd.).")
    ap.add_argument("--out-root", required=True, help="Katalog wyjściowy (zachowa strukturę).")
    ap.add_argument("--workers", type=int, default=0, help="Liczba procesów (0=auto).")
    ap.add_argument("--no-resample", action="store_true",
                    help="Nie resampluj do 16 kHz (błąd jeśli wejście ma inny SR).")
    args = ap.parse_args()

    src = Path(args.in_root).resolve()
    dst = Path(args.out_root).resolve()
    resample = not args.no_resample

    print(f"[cfg] sr_target={SR}, n_mels={N_MELS}, n_fft={N_FFT}, hop={HOP}, frames≈{TARGET_FR}")
    print(f"[run] {src}  →  {dst}  (resample={'yes' if resample else 'no'})")
    stats = run_tree(src, dst, args.workers, resample)

    print("\n=== SUMMARY ===")
    print(f"OK={stats['ok']}  SHORT={stats['short']}  ERRORS={stats['err']}")

if __name__ == "__main__":
    main()
