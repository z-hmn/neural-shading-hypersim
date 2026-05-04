import json
import torch
from torch.utils.data import Dataset


class CachedHypersimDataset(Dataset):
    def __init__(self, cached_index_path, max_samples=None):
        with open(cached_index_path, "r") as f:
            self.samples = json.load(f)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = torch.load(self.samples[idx]["path"], map_location="cpu")
        return item["x"], item["y"], item["reflectance"]
