import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from work.data import *
path = '/data/zbw/course/AAAI/project/AAAI_project'

path_to_data = '/path/to/your/data'
batch_size = 256  # 或您选择的任何适当的批量大小
transform = None
train_data = MinistDataLoader(root_dir= path  + '/processed_data/train/', transform=transform)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

def extract_features(data_loader):
    features = []
    for batch in data_loader:
        samples, _ = batch
        samples = samples.view(samples.size(0), -1)
        features.append(samples.numpy())
    return np.concatenate(features, axis=0)

train_features = extract_features(train_data_loader)

k_values = range(1, 20)  # 比如测试从1到19的k值
wcss = []  # 用于存储每个k值的WCSS

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(train_features)
    wcss.append(kmeans.inertia_)
    print(kmeans.inertia_, k)