from datasets import HypersimIlluminationDataset

ds = HypersimIlluminationDataset("samples_ai_001_001.json", max_samples=10)

print("Dataset length:", len(ds))

x, y = ds[0]

print("Input x shape:", x.shape)
print("Target y shape:", y.shape)
print("x min/max:", x.min().item(), x.max().item())
print("y min/max:", y.min().item(), y.max().item())
