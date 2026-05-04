from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from datasets import HypersimIlluminationDataset
from model import SmallUNet


OUTPUT_DIR = Path("outputs/overfit_one_frame")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

ds = HypersimIlluminationDataset("samples_ai_001_001.json", max_samples=1)
x, y = ds[0]

x = x.unsqueeze(0).to(device)
y = y.unsqueeze(0).to(device)

model = SmallUNet().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = []

EPOCHS = 200

for epoch in range(EPOCHS):
    model.train()

    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | L1 loss: {loss.item():.6f}")

torch.save(model.state_dict(), OUTPUT_DIR / "overfit_one_frame.pt")

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Overfit-One-Frame Diagnostic")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overfit_loss_curve.png", dpi=200)
plt.close()

model.eval()
with torch.no_grad():
    pred = model(x)[0].cpu()
    target = y[0].cpu()
    inp = x[0].cpu()

normals = torch.clamp((inp[0:3] + 1) / 2, 0, 1)
depth = inp[3]
reflectance = inp[4:7]

def img(t):
    return t.permute(1, 2, 0).numpy()

pred_illum = img(pred)
gt_illum = img(target)
reflectance_img = img(reflectance)

pred_recon = np.clip(reflectance_img * pred_illum, 0, 1)
gt_recon = np.clip(reflectance_img * gt_illum, 0, 1)
recon_error = np.abs(pred_recon - gt_recon)

fig, axs = plt.subplots(3, 3, figsize=(14, 14))

# Row 0: inputs
axs[0, 0].imshow(img(normals))
axs[0, 0].set_title("Input Normals")

axs[0, 1].imshow(depth.numpy(), cmap="gray")
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

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overfit_prediction.png", dpi=200)
plt.close()

print("Saved overfit outputs to:", OUTPUT_DIR)