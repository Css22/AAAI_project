import torch
import os
import sys
sys.path.append('/data/zbw/course/AAAI/project/work/')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

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
        
        
        if self.transform:
            sample = self.transform(sample)

        return sample, active_channel, label

path = '/data/zbw/course/AAAI/project/work'
batch_size = 64
train_data = MinistDataLoader(root_dir= path  + '/processed_data/train/')
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

val_data = MinistDataLoader(root_dir= path  +'/processed_data/val/')
val_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(10, 6, 5)  
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModifiedResNet(models.ResNet):
    def __init__(self, block, layers, num_classes=10):
        super(ModifiedResNet, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改为10个通道的输入
        self.fc = nn.Linear(512 * block.expansion, num_classes)



model_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for i in range(0, 10):
    model_list.append(ModifiedResNet(models.resnet.BasicBlock, [2, 2, 2, 2]).to(device))
    # model_list.append(LeNet5().to(device))

optimizer_list = []
for i in range(0, 10):
    optimizer_list.append(optim.SGD(model_list[i].parameters(), lr=0.001, momentum=0.9))

criterion = nn.CrossEntropyLoss()

sub_datasets = {env: [] for env in range(10)}

for i, (data, env, label) in enumerate(train_data_loader):
    for j in range(data.size(0)):
        env_idx = env[j].item()
        sub_datasets[env_idx].append((data[j], label[j]))


class SubDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data_loaders = {}

for env in sub_datasets:
    dataset = SubDataset(sub_datasets[env])
    data_loaders[env] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)






num_epochs = 1


for env_index in range(0, 10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loaders[env_index], 0):

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_list[env_index].zero_grad()

            outputs = model_list[env_index](inputs)
        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_list[env_index].step()

            running_loss += loss.item()
            if i % 100 == 99:   
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0




sub_datasets = {env: [] for env in range(10)}

for i, (data, env, label) in enumerate(val_data_loader):
    for j in range(data.size(0)):
        env_idx = env[j].item()
        sub_datasets[env_idx].append((data[j], label[j]))


for env in sub_datasets:
    dataset = SubDataset(sub_datasets[env])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    data_loaders[env] = data_loader


for i in model_list:
    i.eval()

correct = 0
total = 0
with torch.no_grad():
    for env_index in range(0, 10):
         for i, data in enumerate(data_loaders[env_idx], 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_list[env_idx](images)     
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


val_accuracy = 100 * correct / total
print(val_accuracy)











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
        if self.transform:
            sample = self.transform(sample)

        return sample, active_channel, idx
    

test_data = TestDataLoader(root_dir= path  +'/processed_data/test/')
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
result_list = {}

filename = path + '/result'
correct = 0
total = 0

with open(filename, 'r') as file:
    for line in file:
        parts = line.strip().split(',')        
        result_list[int(parts[0])] = int(parts[1])


with torch.no_grad():
    for i, (data, env, index) in enumerate(test_data_loader):
        for j in range(data.size(0)):
            env_idx = env[j].item()
            input_data = data[j].unsqueeze(0).to(device)
            outputs = model_list[env_idx](input_data)
            _, predicted = torch.max(outputs.data, 1)
            
            total = total + 1
            predicted = predicted.cpu().item()
            if predicted == result_list[index[j].item()]:
                correct = correct + 1
            else:
                print(index[j].item(), predicted, result_list[index[j].item()])


val_accuracy = 100 * correct / total
print(val_accuracy)



