import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import HypersimIlluminationDataset
from model import SmallUNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

ds = HypersimIlluminationDataset("samples_ai_001_001.json", max_samples=20)
loader = DataLoader(ds, batch_size=2, shuffle=True)

model = SmallUNet().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2):
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    print(f"Epoch {epoch+1}, loss: {total / len(loader):.4f}")

torch.save(model.state_dict(), "tiny_unet_checkpoint.pt")
print("Saved tiny_unet_checkpoint.pt")
