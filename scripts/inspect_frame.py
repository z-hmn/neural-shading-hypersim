from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("data/scenes/ai_001_001/images")
FRAME = "frame.0000"

paths = {
    "color": ROOT / "scene_cam_00_final_hdf5" / f"{FRAME}.color.hdf5",
    "reflectance": ROOT / "scene_cam_00_final_hdf5" / f"{FRAME}.diffuse_reflectance.hdf5",
    "illumination": ROOT / "scene_cam_00_final_hdf5" / f"{FRAME}.diffuse_illumination.hdf5",
    "depth": ROOT / "scene_cam_00_geometry_hdf5" / f"{FRAME}.depth_meters.hdf5",
    "normals": ROOT / "scene_cam_00_geometry_hdf5" / f"{FRAME}.normal_bump_cam.hdf5",
}

def load(path):
    with h5py.File(path, "r") as f:
        key = list(f.keys())[0]
        return f[key][:]

def tonemap(x):
    x = np.nan_to_num(x).astype(np.float32)
    x = np.maximum(x, 0)
    return x / (1 + x)

color_raw = load(paths["color"]).astype(np.float32)
reflectance = np.clip(load(paths["reflectance"]).astype(np.float32), 0, 1)
illumination_raw = load(paths["illumination"]).astype(np.float32)
illumination = tonemap(illumination_raw)
color = tonemap(color_raw)
reconstructed = tonemap(reflectance * illumination_raw)

depth = load(paths["depth"]).astype(np.float32)
normals = load(paths["normals"]).astype(np.float32)


depth = np.nan_to_num(depth)
depth = np.log1p(np.maximum(depth, 0))
depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

normals_vis = np.clip((normals + 1) / 2, 0, 1)

fig, axs = plt.subplots(2, 3, figsize=(14, 8))

axs[0, 0].imshow(reflectance)
axs[0, 0].set_title("Diffuse Reflectance / Albedo")

axs[0, 1].imshow(illumination)
axs[0, 1].set_title("Diffuse Illumination")

axs[0, 2].imshow(reconstructed)
axs[0, 2].set_title("Reflectance × Illumination")

axs[1, 0].imshow(color)
axs[1, 0].set_title("Ground Truth Color")

axs[1, 1].imshow(depth, cmap="gray")
axs[1, 1].set_title("Depth Meters")

axs[1, 2].imshow(normals_vis)
axs[1, 2].set_title("Normal Bump Cam")

for ax in axs.ravel():
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/reconstruction_debug.png", dpi=200)
plt.show()

print("Saved outputs/reconstruction_debug.png")


