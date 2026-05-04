from pathlib import Path
import json

SCENES_ROOT = Path("data/scenes")
OUT_PATH = "samples_multiscene_normalWorld.json"

samples = []

for scene_dir in sorted(SCENES_ROOT.glob("ai_*")):
    images_root = scene_dir / "images"

    if not images_root.exists():
        continue

    final_dirs = sorted(images_root.glob("scene_cam_*_final_hdf5"))

    for final_dir in final_dirs:
        cam_name = final_dir.name.replace("_final_hdf5", "")
        geom_dir = images_root / f"{cam_name}_geometry_hdf5"

        if not geom_dir.exists():
            continue

        for refl_path in sorted(final_dir.glob("frame.*.diffuse_reflectance.hdf5")):
            frame = refl_path.name.replace(".diffuse_reflectance.hdf5", "")

            sample = {
                "scene": scene_dir.name,
                "camera": cam_name,
                "frame": frame,
                "reflectance": str(final_dir / f"{frame}.diffuse_reflectance.hdf5"),
                "illumination": str(final_dir / f"{frame}.diffuse_illumination.hdf5"),
                "color": str(final_dir / f"{frame}.color.hdf5"),
                "depth": str(geom_dir / f"{frame}.depth_meters.hdf5"),
                "normals": str(geom_dir / f"{frame}.normal_world.hdf5"),
            }

            required_paths = [
                sample["reflectance"],
                sample["illumination"],
                sample["color"],
                sample["depth"],
                sample["normals"],
            ]

            if all(Path(p).exists() for p in required_paths):
                samples.append(sample)

with open(OUT_PATH, "w") as f:
    json.dump(samples, f, indent=2)

print(f"Found {len(samples)} complete samples across scenes.")
print(f"Saved {OUT_PATH}")

scene_counts = {}
for s in samples:
    scene_counts[s["scene"]] = scene_counts.get(s["scene"], 0) + 1

print("Scene counts:")
for scene, count in scene_counts.items():
    print(scene, count)