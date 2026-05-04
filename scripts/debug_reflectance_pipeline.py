import json
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from cached_dataset import CachedHypersimDataset


RAW_INDEX = "samples_multiscene_normalCam.json"
CACHE_INDEX = "cached_data/normal_cam/cached_index.json"
OUT_DIR = Path("outputs/debug_reflectance")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hdf5(path):
    with h5py.File(path, "r") as f:
        key = list(f.keys())[0]
        arr = f[key][:]
    return arr.astype(np.float32)


def tonemap(x):
    x = np.nan_to_num(x).astype(np.float32)
    x = np.maximum(x, 0)
    return x / (1 + x)


def chw_to_hwc(t):
    return t.detach().cpu().permute(1, 2, 0).numpy()


def print_stats(name, arr):
    arr = np.nan_to_num(arr)
    print(f"\n{name}")
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)
    print("  min/max:", arr.min(), arr.max())
    if arr.ndim == 3:
        print("  channel means RGB:", arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean())
        print("  channel max RGB:", arr[..., 0].max(), arr[..., 1].max(), arr[..., 2].max())


with open(RAW_INDEX, "r") as f:
    raw_samples = json.load(f)

cached_ds = CachedHypersimDataset(CACHE_INDEX)

# test several samples, including ones likely used in your prediction examples
test_indices = [0, 1, 2, 10, 25, 50, 100, 250, 500, 1000]

for idx in test_indices:
    if idx >= len(cached_ds):
        continue

    print("\n" + "=" * 80)
    print("INDEX:", idx)

    raw = raw_samples[idx]
    print("RAW SAMPLE:")
    for k, v in raw.items():
        print(f"  {k}: {v}")

    x, y, cached_refl = cached_ds[idx]

    raw_color = load_hdf5(raw["color"])
    raw_refl = load_hdf5(raw["reflectance"])
    raw_illum = load_hdf5(raw["illumination"])

    # display versions
    raw_color_vis = tonemap(raw_color)
    raw_refl_vis = np.clip(np.nan_to_num(raw_refl), 0, 1)
    raw_illum_vis = tonemap(raw_illum)

    x_normals = torch.clamp((x[0:3] + 1) / 2, 0, 1)
    x_depth = x[3]
    x_refl = x[4:7]

    cached_refl_vis = chw_to_hwc(cached_refl)
    x_refl_vis = chw_to_hwc(x_refl)
    y_vis = chw_to_hwc(y)

    print_stats("RAW COLOR tone-mapped", raw_color_vis)
    print_stats("RAW REFLECTANCE clipped", raw_refl_vis)
    print_stats("RAW ILLUM tone-mapped", raw_illum_vis)
    print_stats("CACHED reflectance", cached_refl_vis)
    print_stats("X reflectance x[4:7]", x_refl_vis)
    print_stats("Y cached illumination", y_vis)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    axs[0, 0].imshow(raw_color_vis)
    axs[0, 0].set_title("RAW color")

    axs[0, 1].imshow(raw_refl_vis)
    axs[0, 1].set_title("RAW reflectance")

    axs[0, 2].imshow(raw_illum_vis)
    axs[0, 2].set_title("RAW illumination")

    axs[0, 3].imshow(np.clip(raw_refl_vis * raw_illum_vis, 0, 1))
    axs[0, 3].set_title("RAW refl × RAW illum")

    axs[1, 0].imshow(chw_to_hwc(x_normals))
    axs[1, 0].set_title("Cached x normals")

    axs[1, 1].imshow(x_depth.numpy(), cmap="gray")
    axs[1, 1].set_title("Cached x depth")

    axs[1, 2].imshow(x_refl_vis)
    axs[1, 2].set_title("Cached x[4:7]")

    axs[1, 3].imshow(cached_refl_vis)
    axs[1, 3].set_title("Cached reflectance item")

    for ax in axs.ravel():
        ax.axis("off")

    plt.suptitle(f"Reflectance Debug idx={idx} scene={raw.get('scene')} frame={raw.get('frame')}")
    plt.tight_layout()

    out = OUT_DIR / f"debug_reflectance_idx_{idx}.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print("Saved:", out)

print("\nDONE.")
