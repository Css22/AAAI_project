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
from torchvision import  transforms

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
    
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)  
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1 ,padding=0)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.avg_pool2d(x,2,2))
        x = self.conv2(x)
        x = F.relu(F.avg_pool2d(x,2,2))

        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x



class ModifiedResNet(models.ResNet):
    def __init__(self, block, layers, num_classes=10):
        super(ModifiedResNet, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改为10个通道的输入
        self.fc = nn.Linear(512 * block.expansion, num_classes)



transform = transforms.Compose([
    transforms.Resize(224),  # 放大图像
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])
transform = None
path = '/data/zbw/course/AAAI/project/work'
batch_size = 32
train_data = MinistDataLoader(root_dir= path  + '/processed_data/train/', transform=transform)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

val_data = MinistDataLoader(root_dir= path  +'/processed_data/val/', transform=transform)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

test_data = TestDataLoader(root_dir= path  +'/processed_data/test/', transform=transform)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

# lenet5_model = LeNet5()
# lenet5_model.train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Training on device: {device}")
# lenet5_model.to(device)

# criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(lenet5_model.parameters(), lr=0.01, momentum=0.9)


model = ModifiedResNet(models.resnet.BasicBlock, [2, 2, 2, 2])
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 20


for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:   
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

model.eval()
correct = 0
total = 0
with torch.no_grad(): 
    for i, data in enumerate(val_data_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)     
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


val_accuracy = 100 * correct / total

print(val_accuracy)

filename = path + '/result'
with torch.no_grad():
    with open(filename, 'w') as file:
        for i, data in enumerate(test_data_loader, 0):
            images, index = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            index = index.cpu().numpy()
            predicted = predicted.cpu().numpy()
            for i in range(0, len(index)):
                file.write( str(index[i])+ "," + str(predicted[i])+'\n')
                

