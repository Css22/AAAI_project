import torch
import sys
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append('/data/zbw/course/AAAI/project/AAAI_project')

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
        
        sample = torch.from_numpy(sample)

        active_channel = None
        for i, channel in enumerate(sample):
            if torch.count_nonzero(channel) > 0:
                active_channel = i
                break
        sample = sample[active_channel].unsqueeze(0)
        if self.transform:
            sample = self.transform(sample)

        return sample , label

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
        sample = torch.from_numpy(sample)
        active_channel = None
        for i, channel in enumerate(sample):
            if torch.count_nonzero(channel) > 0:
                active_channel = i
                break
        sample = sample[active_channel].unsqueeze(0)
   
        if self.transform:
            sample = self.transform(sample)

        return sample, idx