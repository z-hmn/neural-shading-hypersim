import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import HypersimIlluminationDataset, load_hdf5, tonemap, resize_array
from model import SmallUNet


INDEX_PATH = "samples_multiscene_normalCam.json"
OUTPUT_DIR = Path("outputs/aws_normalCam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
MAX_SAMPLES = None  # use all 99 samples

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ----------------------------
# 1. Load dataset
# ----------------------------

dataset = HypersimIlluminationDataset(
    INDEX_PATH,
    image_size=IMAGE_SIZE,
    max_samples=MAX_SAMPLES,
)

n = len(dataset)
indices = list(range(n))

random.seed(42)
random.shuffle(indices)

train_size = int(0.8 * n)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

test_loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)


print(f"Total samples: {n}")
print(f"Train samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")


# ----------------------------
# 2. Create model/loss/optimizer
# ----------------------------

model = SmallUNet().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_losses = []


# ----------------------------
# 3. Evaluation helper
# ----------------------------

def evaluate(loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

    return total_loss / len(loader)


# ----------------------------
# 4. Training loop
# ----------------------------

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_test_loss = evaluate(test_loader)

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train L1: {avg_train_loss:.4f} | "
        f"test L1: {avg_test_loss:.4f}"
    )


# ----------------------------
# 5. Save checkpoint
# ----------------------------

checkpoint_path = OUTPUT_DIR / "small_unet_train_eval.pt"
torch.save(model.state_dict(), checkpoint_path)
print(f"Saved checkpoint: {checkpoint_path}")


# ----------------------------
# 6. Save loss curve
# ----------------------------

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train L1")
plt.plot(test_losses, label="Test L1")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.tight_layout()
loss_curve_path = OUTPUT_DIR / "loss_curve.png"
plt.savefig(loss_curve_path, dpi=200)
plt.close()

print(f"Saved loss curve: {loss_curve_path}")


# ----------------------------
# 7. Helpers for visualization
# ----------------------------

def tensor_to_img(t):
    """
    Convert tensor from [C, H, W] to numpy image [H, W, C].
    """
    return t.detach().cpu().permute(1, 2, 0).numpy()


def tensor_to_gray(t):
    """
    Convert tensor from [H, W] or [1, H, W] to numpy grayscale image.
    """
    t = t.detach().cpu()
    if t.ndim == 3:
        t = t[0]
    return t.numpy()


def load_reflectance_from_sample(sample):
    reflectance = load_hdf5(sample["reflectance"])
    reflectance = resize_array(reflectance, IMAGE_SIZE)
    reflectance = np.clip(np.nan_to_num(reflectance), 0, 1)
    return reflectance.astype(np.float32)


# ----------------------------
# 8. Save qualitative predictions
# ----------------------------

model.eval()

with open(INDEX_PATH, "r") as f:
    all_samples = json.load(f)

num_examples_to_save = min(5, len(test_indices))

with torch.no_grad():
    for i in range(num_examples_to_save):
        dataset_idx = test_indices[i]
        sample = all_samples[dataset_idx]

        x, y = dataset[dataset_idx]
        x_batch = x.unsqueeze(0).to(device)

        pred = model(x_batch)[0].cpu()

        # Pull individual components from x
        normals = x[0:3]
        depth = x[3]
        reflectance_tensor = x[4:7]

        normals_vis = torch.clamp((normals + 1) / 2, 0, 1)
        depth_vis = depth
        reflectance = tensor_to_img(reflectance_tensor)

        gt_illum = tensor_to_img(y)
        pred_illum = tensor_to_img(pred)

        # Reconstruct color using reflectance × illumination.
        # These are both tone-mapped-ish display-space images, so this is not physically exact,
        # but it is useful as a qualitative visualization.
        gt_recon = np.clip(reflectance * gt_illum, 0, 1)
        pred_recon = np.clip(reflectance * pred_illum, 0, 1)

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))

        axs[0, 0].imshow(tensor_to_img(normals_vis))
        axs[0, 0].set_title("Input Normals")

        axs[0, 1].imshow(tensor_to_gray(depth_vis), cmap="gray")
        axs[0, 1].set_title("Input Depth")

        axs[0, 2].imshow(reflectance)
        axs[0, 2].set_title("Input Reflectance")

        axs[0, 3].imshow(gt_illum)
        axs[0, 3].set_title("GT Illumination")

        axs[1, 0].imshow(pred_illum)
        axs[1, 0].set_title("Predicted Illumination")

        axs[1, 1].imshow(np.abs(pred_illum - gt_illum))
        axs[1, 1].set_title("Illumination Error")

        axs[1, 2].imshow(gt_recon)
        axs[1, 2].set_title("GT Reflectance × Illum")

        axs[1, 3].imshow(pred_recon)
        axs[1, 3].set_title("Pred Reflectance × Illum")

        for ax in axs.ravel():
            ax.axis("off")

        plt.suptitle(f"Test Sample: {sample['frame']}")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"prediction_example_{i}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved prediction example: {out_path}")


print("Done.")