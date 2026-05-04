import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_hdf5(path):
    with h5py.File(path, "r") as f:
        key = list(f.keys())[0]
        arr = f[key][:]
    return arr.astype(np.float32)


def resize_array(arr, size=(256, 256)):
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        img = img.resize(size, Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    channels = []
    for c in range(arr.shape[2]):
        img = Image.fromarray(arr[..., c].astype(np.float32), mode="F")
        img = img.resize(size, Image.BILINEAR)
        channels.append(np.array(img, dtype=np.float32))

    return np.stack(channels, axis=-1)


def tonemap(x):
    x = np.nan_to_num(x).astype(np.float32)
    x = np.maximum(x, 0)
    return x / (1 + x)


class HypersimIlluminationDataset(Dataset):
    def __init__(self, index_path, image_size=(256, 256), max_samples=None):
        with open(index_path, "r") as f:
            self.samples = json.load(f)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        reflectance = load_hdf5(sample["reflectance"])
        illumination = load_hdf5(sample["illumination"])
        depth = load_hdf5(sample["depth"])
        normals = load_hdf5(sample["normals"])

        reflectance = resize_array(reflectance, self.image_size)
        illumination = resize_array(illumination, self.image_size)
        depth = resize_array(depth, self.image_size)
        normals = resize_array(normals, self.image_size)

        reflectance = np.clip(np.nan_to_num(reflectance), 0, 1)
        illumination = tonemap(illumination)

        depth = np.nan_to_num(depth)
        depth = np.log1p(np.maximum(depth, 0))
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        normals = np.nan_to_num(normals)
        normals = np.clip(normals, -1, 1)

        x = np.concatenate(
            [
                normals,
                depth[..., None],
                reflectance,
            ],
            axis=-1,
        )

        y = illumination

        x = torch.from_numpy(x).permute(2, 0, 1).float()
        y = torch.from_numpy(y).permute(2, 0, 1).float()

        return x, y
