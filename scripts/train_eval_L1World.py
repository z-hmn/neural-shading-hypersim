import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cached_dataset import CachedHypersimDataset
from model import SmallUNet


CACHE_INDEX = "cached_data/normal_world/cached_index.json"
OUTPUT_DIR = Path("outputs/exp1_normalWorld_illumLoss_20epochs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 14
LEARNING_RATE = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

dataset = CachedHypersimDataset(CACHE_INDEX)

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
    num_workers=4,
    pin_memory=True,
)

test_loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"Total samples: {n}")
print(f"Train samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")

model = SmallUNet().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


train_losses = []
test_losses = []




def evaluate(loader):
   model.eval()
   total_loss = 0.0


   with torch.no_grad():
       for x, y, reflectance in loader:
           x = x.to(device)
           y = y.to(device)


           pred = model(x)
           loss = loss_fn(pred, y)
           total_loss += loss.item()


   return total_loss / len(loader)




for epoch in range(EPOCHS):
   model.train()
   total_train_loss = 0.0


   for x, y, reflectance in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
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


checkpoint_path = OUTPUT_DIR / "small_unet_train_eval.pt"
torch.save(model.state_dict(), checkpoint_path)
print(f"Saved checkpoint: {checkpoint_path}")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train L1")
plt.plot(test_losses, label="Test L1")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Normal World: Illumination and Reconstruction Loss")
plt.legend()
plt.tight_layout()
loss_curve_path = OUTPUT_DIR / "loss_curve.png"
plt.savefig(loss_curve_path, dpi=200)
plt.close()
print(f"Saved loss curve: {loss_curve_path}")


def tensor_to_img(t):
    return t.detach().cpu().permute(1, 2, 0).numpy()


def tensor_to_gray(t):
    t = t.detach().cpu()
    if t.ndim == 3:
        t = t[0]
    return t.numpy()


model.eval()
num_examples_to_save = min(5, len(test_indices))

with torch.no_grad():
    for i in range(num_examples_to_save):
        dataset_idx = test_indices[i]

        x, y, reflectance = dataset[dataset_idx]
        x_batch = x.unsqueeze(0).to(device)

        pred = model(x_batch)[0].cpu()

        normals = x[0:3]
        depth = x[3]

        normals_vis = torch.clamp((normals + 1) / 2, 0, 1)
        depth_vis = depth

        reflectance_img = tensor_to_img(reflectance)
        gt_illum = tensor_to_img(y)
        pred_illum = tensor_to_img(pred)

        gt_recon = np.clip(reflectance_img * gt_illum, 0, 1)
        pred_recon = np.clip(reflectance_img * pred_illum, 0, 1)
        recon_error = np.abs(pred_recon - gt_recon)

        fig, axs = plt.subplots(3, 3, figsize=(14, 14))

        # Row 0: inputs
        axs[0, 0].imshow(tensor_to_img(normals_vis))
        axs[0, 0].set_title("Input Normals")

        axs[0, 1].imshow(tensor_to_gray(depth_vis), cmap="gray")
        axs[0, 1].set_title("Input Depth")

        axs[0, 2].imshow(reflectance_img)
        axs[0, 2].set_title("Input Reflectance")

        # Row 1: illumination
        axs[1, 0].imshow(gt_illum)
        axs[1, 0].set_title("GT Illumination")

        axs[1, 1].imshow(pred_illum)
        axs[1, 1].set_title("Predicted Illumination")

        axs[1, 2].imshow(np.abs(pred_illum - gt_illum))
        axs[1, 2].set_title("Illumination Error")

        # Row 2: reconstruction
        axs[2, 0].imshow(gt_recon)
        axs[2, 0].set_title("GT Reflectance × Illum")

        axs[2, 1].imshow(pred_recon)
        axs[2, 1].set_title("Pred Reflectance × Illum")

        axs[2, 2].imshow(recon_error)
        axs[2, 2].set_title("Reconstruction Error")

        for ax in axs.ravel():
            ax.axis("off")

        plt.suptitle(f"Normal World Test Example {i}")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"prediction_example_{i}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved prediction example: {out_path}")

print("Done.")
