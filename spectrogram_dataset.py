import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class NpySpectrogramDataset(Dataset):
    def __init__(self, root, mel_bins=80, expected_width=None, class_to_label=None,
        aug_ratio=0.0, use_noise=False, use_chop=False, use_fast=False, use_slow=False):
        random.seed(300)

        self.root = root
        self.mel_bins = mel_bins
        self.expected_width = expected_width

        if class_to_label is None:
            self.class_to_label = {"class_0": 0, "class_1": 1}
        else:
            self.class_to_label = class_to_label

        original_samples = []
        aug_samples = []

        for class_name, label in self.class_to_label.items():
            class_dir = os.path.join(root, class_name)
            for speaker in os.listdir(class_dir):
                speaker_dir = os.path.join(class_dir, speaker)
                for filename in os.listdir(speaker_dir):
                    if filename.endswith(".npy"):
                        full_path = os.path.join(speaker_dir, filename)
                        if filename.endswith("_noise.npy"):
                            if use_noise:
                                aug_samples.append((full_path, label))
                        elif filename.endswith("_chop.npy"):
                            if use_chop:
                                aug_samples.append((full_path, label))
                        elif filename.endswith("_fast.npy"):
                            if use_fast:
                                aug_samples.append((full_path, label))
                        elif filename.endswith("_slow.npy"):
                            if use_slow:
                                aug_samples.append((full_path, label))
                        else:
                            original_samples.append((full_path, label))

        if 0.0 <= aug_ratio < 1.0 and len(aug_samples) > 0:
            k = int(round(len(aug_samples) * aug_ratio))
            aug_samples = random.sample(aug_samples, k)

        self.samples = original_samples + aug_samples
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        arr = np.load(path)

        if arr.ndim != 2:
            arr = np.squeeze(arr)

        H, W = arr.shape
        if H != self.mel_bins:
            raise ValueError(f"{path} mel bins {H}, expected {self.mel_bins}")
        if W != self.expected_width:
            raise ValueError(f"{path} width {W}, expected {self.expected_width}")
        
        x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # [1, 80, W]
        y = torch.tensor(label, dtype=torch.long)
        return x, y