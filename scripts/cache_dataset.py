import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


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


def preprocess_sample(sample, image_size):
    reflectance = load_hdf5(sample["reflectance"])
    illumination = load_hdf5(sample["illumination"])
    depth = load_hdf5(sample["depth"])
    normals = load_hdf5(sample["normals"])

    reflectance = resize_array(reflectance, image_size)
    illumination = resize_array(illumination, image_size)
    depth = resize_array(depth, image_size)
    normals = resize_array(normals, image_size)

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
    reflectance_tensor = torch.from_numpy(reflectance).permute(2, 0, 1).float()

    return {
        "x": x,
        "y": y,
        "reflectance": reflectance_tensor,
        "frame": sample.get("frame", ""),
        "scene": sample.get("scene", ""),
        "camera": sample.get("camera", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.index, "r") as f:
        samples = json.load(f)

    cached_index = []

    for i, sample in enumerate(tqdm(samples, desc="Caching samples")):
        item = preprocess_sample(sample, (args.image_size, args.image_size))

        out_path = out_dir / f"sample_{i:06d}.pt"
        torch.save(item, out_path)

        cached_index.append(
            {
                "path": str(out_path),
                "frame": item["frame"],
                "scene": item["scene"],
                "camera": item["camera"],
            }
        )

    index_out = out_dir / "cached_index.json"
    with open(index_out, "w") as f:
        json.dump(cached_index, f, indent=2)

    print(f"Cached {len(cached_index)} samples.")
    print(f"Saved cached index to {index_out}")


if __name__ == "__main__":
    main()

