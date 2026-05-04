from pathlib import Path
import json

ROOT = Path("data/scenes/ai_001_001/images")
FINAL = ROOT / "scene_cam_00_final_hdf5"
GEOM = ROOT / "scene_cam_00_geometry_hdf5"

samples = []

for refl_path in sorted(FINAL.glob("frame.*.diffuse_reflectance.hdf5")):
    frame = refl_path.name.replace(".diffuse_reflectance.hdf5", "")

    sample = {
        "frame": frame,
        "reflectance": str(FINAL / f"{frame}.diffuse_reflectance.hdf5"),
        "illumination": str(FINAL / f"{frame}.diffuse_illumination.hdf5"),
        "color": str(FINAL / f"{frame}.color.hdf5"),
        "depth": str(GEOM / f"{frame}.depth_meters.hdf5"),
        "normals": str(GEOM / f"{frame}.normal_bump_cam.hdf5"),
    }

    if all(Path(p).exists() for p in sample.values() if p != frame):
        samples.append(sample)

with open("samples_ai_001_001.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"Found {len(samples)} complete samples.")
print("Saved samples_ai_001_001.json")
print(samples[0])
