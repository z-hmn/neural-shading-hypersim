import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets import HypersimIlluminationDataset
from model import SmallUNet


INDEX_PATH = "samples_ai_001_001.json"
OUTPUT_DIR = Path("outputs/train_eval_gradient_loss")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2
EPOCHS = 8
LR = 1e-4
GRAD_WEIGHT = 0.25

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

dataset = HypersimIlluminationDataset(INDEX_PATH)

n = len(dataset)
indices = list(range(n))
random.seed(42)
random.shuffle(indices)

train_size = int(0.8 * n)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

model = SmallUNet().to(device)
l1 = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses = []
test_losses = []


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    return l1(pred_dx, target_dx) + l1(pred_dy, target_dy)


def total_loss(pred, target):
    return l1(pred, target) + GRAD_WEIGHT * gradient_loss(pred, target)


def evaluate(loader):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = total_loss(pred, y)
            total += loss.item()

    return total / len(loader)


for epoch in range(EPOCHS):
    model.train()
    total = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = total_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    train_loss = total / len(train_loader)
    test_loss = evaluate(test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"train combined loss: {train_loss:.4f} | "
        f"test combined loss: {test_loss:.4f}"
    )

torch.save(model.state_dict(), OUTPUT_DIR / "small_unet_gradient_loss.pt")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.xlabel("Epoch")
plt.ylabel("L1 + Gradient Loss")
plt.title("Training with Gradient Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve_gradient.png", dpi=200)
plt.close()


def img(t):
    return t.detach().cpu().permute(1, 2, 0).numpy()


model.eval()

num_examples = min(5, len(test_indices))

with torch.no_grad():
    for i in range(num_examples):
        dataset_idx = test_indices[i]
        x, y = dataset[dataset_idx]

        pred = model(x.unsqueeze(0).to(device))[0].cpu()

        normals = torch.clamp((x[0:3] + 1) / 2, 0, 1)
        depth = x[3]
        reflectance = x[4:7]

        pred_illum = img(pred)
        gt_illum = img(y)
        reflectance_img = img(reflectance)

        pred_recon = np.clip(reflectance_img * pred_illum, 0, 1)
        gt_recon = np.clip(reflectance_img * gt_illum, 0, 1)

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))

        axs[0, 0].imshow(img(normals))
        axs[0, 0].set_title("Input Normals")

        axs[0, 1].imshow(depth.numpy(), cmap="gray")
        axs[0, 1].set_title("Input Depth")

        axs[0, 2].imshow(reflectance_img)
        axs[0, 2].set_title("Input Reflectance")

        axs[0, 3].imshow(gt_illum)
        axs[0, 3].set_title("GT Illumination")

        axs[1, 0].imshow(pred_illum)
        axs[1, 0].set_title("Pred Illumination")

        axs[1, 1].imshow(np.abs(pred_illum - gt_illum))
        axs[1, 1].set_title("Error")

        axs[1, 2].imshow(gt_recon)
        axs[1, 2].set_title("GT Recon")

        axs[1, 3].imshow(pred_recon)
        axs[1, 3].set_title("Pred Recon")

        for ax in axs.ravel():
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"prediction_gradient_{i}.png", dpi=200)
        plt.close()

        print(f"Saved example {i}")

print("Done.")