import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TestDataLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = []
        self.transform = transform
        for file in os.listdir(root_dir):
            self.samples.append(os.path.join(root_dir, file))
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        sample = np.load(file_path)

        if self.transform:
            sample = self.transform(sample)

        return torch.from_numpy(sample), idx
    


class MinistDataLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = []
        self.transform = transform

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            for file in os.listdir(label_path):
                self.samples.append((os.path.join(label_path, file), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        sample = np.load(file_path)

        if self.transform:
            sample = self.transform(sample)

        return torch.from_numpy(sample), label

# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

