import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import librosa

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
FMIN, FMAX = 20, 7600
SEG_SEC = 3.0
TARGET_FR = int(round(SEG_SEC * SR / HOP))  # ~300 frames


AUDIO_GLOB = "*.wav"


def compute_logmel_db(y, sr=SR):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0
    )
    # log-Mel in dB
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def center_trim_or_pad(M, target_frames=TARGET_FR):
    n_mels, T = M.shape
    if T == target_frames:
        return M
    if T > target_frames:
        s = (T - target_frames) // 2
        return M[:, s:s + target_frames]
    return np.pad(M, ((0, 0), (0, target_frames - T)), mode="constant")


def load_wav_mono(path: Path, resample: bool):
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)

    if resample and sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    elif (not resample) and sr != SR:
        raise ValueError(f"SR={sr} but resampling disabled")

    return y, sr

def norm_none(M, stats=None):
    return M

def norm_cmvn(M, stats=None):
    mu = float(M.mean())
    std = float(M.std()) + 1e-10
    return (M - mu) / std

def norm_minmax(M, stats=None):
    mn = float(M.min())
    mx = float(M.max())
    return (M - mn) / ((mx - mn) + 1e-10)

def norm_robust(M, stats=None):
    p5 = float(np.percentile(M, 5))
    p95 = float(np.percentile(M, 95))
    M = np.clip(M, p5, p95)
    return (M - p5) / ((p95 - p5) + 1e-10)

def norm_global(M, stats):
    # stats: {"mu": float, "std": float}
    return (M - stats["mu"]) / stats["std"]

def norm_freq_global(M, stats):
    # stats: {"mu_f": [n_mels,1], "std_f": [n_mels,1]}
    return (M - stats["mu_f"]) / stats["std_f"]


NORM_FUNCS = {
    "none": norm_none,
    "cmvn": norm_cmvn,
    "minmax": norm_minmax,
    "robust": norm_robust,
    "global": norm_global,
    "freq_global": norm_freq_global,
}


def list_wavs(root: Path):
    return sorted(root.rglob(AUDIO_GLOB))


def compute_global_stats(train_root: Path, resample: bool, mode: str):
    files = list_wavs(train_root)
    if not files:
        raise RuntimeError(f"No wavs found in train root: {train_root}")

    if mode == "global":
        total_sum = 0.0
        total_sq = 0.0
        total_count = 0

        for p in files:
            y, sr = load_wav_mono(p, resample=resample)
            if len(y) < int(0.5 * SR):
                continue
            M = center_trim_or_pad(compute_logmel_db(y, sr), TARGET_FR)
            total_sum += float(M.sum())
            total_sq += float((M * M).sum())
            total_count += int(M.size)

        mu = total_sum / total_count
        var = (total_sq / total_count) - mu * mu
        std = float(np.sqrt(var + 1e-10))
        return {"mu": float(mu), "std": std}

    if mode == "freq_global":
        # per-frequency mean/std over (all samples, all time frames)
        # accumulate sum and sumsq for each mel bin
        sum_f = np.zeros((N_MELS, 1), dtype=np.float64)
        sq_f = np.zeros((N_MELS, 1), dtype=np.float64)
        count = 0  # number of time frames accumulated across all samples (after trim/pad)

        for p in files:
            y, sr = load_wav_mono(p, resample=resample)
            if len(y) < int(0.5 * SR):
                continue
            M = center_trim_or_pad(compute_logmel_db(y, sr), TARGET_FR)  # [n_mels, T]
            sum_f += M.mean(axis=1, keepdims=True) * M.shape[1]
            sq_f += (M * M).mean(axis=1, keepdims=True) * M.shape[1]
            count += M.shape[1]

        mu_f = sum_f / count
        var_f = (sq_f / count) - (mu_f * mu_f)
        std_f = np.sqrt(var_f + 1e-10)
        return {"mu_f": mu_f.astype(np.float32), "std_f": std_f.astype(np.float32)}

    raise ValueError("mode must be 'global' or 'freq_global'")


def process_one(wav_path: Path, in_root: Path, out_root: Path, resample: bool, norm: str, stats):
    rel = wav_path.relative_to(in_root)
    out_base = (out_root / rel).with_suffix("")  # .npy later
    out_base.parent.mkdir(parents=True, exist_ok=True)

    try:
        y, sr = load_wav_mono(wav_path, resample=resample)
        if len(y) < int(0.5 * SR):
            return "short", str(rel)

        M = compute_logmel_db(y, sr)
        M = center_trim_or_pad(M, TARGET_FR)

        M = NORM_FUNCS[norm](M, stats)

        np.save(out_base.with_suffix(".npy"), M.astype(np.float32))
        return "ok", str(rel)

    except Exception as e:
        return "err", f"{rel} :: {type(e).__name__}: {e}"


def run_all(in_root: Path, out_root: Path, resample: bool, norm: str, stats, workers: int):
    files = list_wavs(in_root)
    if not files:
        raise RuntimeError(f"No wavs found in: {in_root}")

    stats_count = {"ok": 0, "err": 0, "short": 0}
    max_workers = workers or os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(process_one, p, in_root, out_root, resample, norm, stats)
            for p in files
        ]
        for f in as_completed(futs):
            status, msg = f.result()
            stats_count[status] += 1
            if status != "ok" and stats_count[status] <= 10:
                print(f"[{status.upper()}] {msg}")

    return stats_count


def main():
    ap = argparse.ArgumentParser(description="Generate log-Mel spectrograms with chosen input normalization (no model changes).")
    ap.add_argument("--in-root", required=True, help="np. raw_data")
    ap.add_argument("--out-root", required=True, help="np. features_logmel_global")
    ap.add_argument("--norm", choices=list(NORM_FUNCS.keys()), default="none",
                    help="Normalization applied on spectrograms.")
    ap.add_argument("--workers", type=int, default=0, help="0=auto")
    ap.add_argument("--no-resample", action="store_true")
    ap.add_argument("--train-subdir", default="train_data", help="subfolder name containing training data")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    resample = not args.no_resample

    train_root = in_root / args.train_subdir
    if not train_root.exists():
        raise FileNotFoundError(f"Train folder not found: {train_root}")

    # Compute stats if needed
    stats = None
    if args.norm in ("global", "freq_global"):
        print(f"Computing {args.norm} stats from TRAIN only: {train_root}")
        stats = compute_global_stats(train_root, resample=resample, mode=args.norm)
        out_root.mkdir(parents=True, exist_ok=True)
        np.savez(out_root / "norm_stats_train.npz", **stats)
        if args.norm == "global":
            print(f"[STATS] mu={stats['mu']:.4f}, std={stats['std']:.4f}")
        else:
            print(f"[STATS] saved per-frequency stats to {out_root/'norm_stats_train.npz'}")

    print(f"Generating features for ALL splits under: {in_root}")
    print(f"norm={args.norm}, resample={'yes' if resample else 'no'} → {out_root}")

    result = run_all(in_root, out_root, resample=resample, norm=args.norm, stats=stats, workers=args.workers)
    print("\n=== SUMMARY ===")
    print(result)


if __name__ == "__main__":
    main()
