from datasets import HypersimIlluminationDataset
from model import SmallUNet

ds = HypersimIlluminationDataset("samples_ai_001_001.json", max_samples=2)
x, y = ds[0]

model = SmallUNet()

x_batch = x.unsqueeze(0)
pred = model(x_batch)

print("Input batch:", x_batch.shape)
print("Prediction:", pred.shape)
print("Target:", y.unsqueeze(0).shape)
