import sys

sys.path.append('/data/zbw/course/AAAI/project/AAAI_project')
import torchvision.models as models
from torchvision import  transforms
import torch.nn.functional as F
import torch.optim as optim
from model.util import *
path = '/data/zbw/course/AAAI/project/work'



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, in_dim, name=None):
        self.inplanes = 64
        self.in_dim = in_dim
            
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(self.in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_classes = 10
in_dim = 1
NUM_EPOCHS = 10

model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   in_dim = in_dim)


model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


transform = transforms.Compose([
    transforms.Resize(224),  # 放大图像
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])
transform = None

train_data = MinistDataLoader(root_dir= path  + '/processed_data/train/', transform=transform)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

val_data = MinistDataLoader(root_dir= path  +'/processed_data/val/', transform=transform)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

test_data = TestDataLoader(root_dir= path  +'/processed_data/test/', transform=transform)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)


# for epoch in range(NUM_EPOCHS):
#     model.train()
#     for batch_idx, (features, targets) in enumerate(train_data_loader):
    
#         ### PREPARE MINIBATCH
#         features = features.to(DEVICE)
#         targets = targets.to(DEVICE)
            
#         ### FORWARD AND BACK PROP
#         logits, probas = model(features)
#         cost = F.cross_entropy(logits, targets)
#         optimizer.zero_grad()
        
#         cost.backward()
        
#         ### UPDATE MODEL PARAMETERS
#         optimizer.step()
        
#         ### LOGGING
#         if not batch_idx % 200:
#             print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
#                    f'Batch {batch_idx:04d}/{len(train_data_loader):04d} |' )
            


# model.eval()
# correct = 0
# total = 0
# with torch.no_grad(): 
#     for i, data in enumerate(val_data_loader, 0):
#         images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
#         logits, probas = model(images)
#         _, predicted = torch.max(logits, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()


# val_accuracy = 100 * correct / total

# print(val_accuracy)

import time
from helper import compute_accuracy_and_loss

start_time = time.time()
train_acc_lst, valid_acc_lst = [], []
train_loss_lst, valid_loss_lst = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_data_loader):
    
        ### PREPARE MINIBATCH
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 200:
            print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                   f'Batch {batch_idx:04d}/{len(train_data_loader):04d} |' 
                   f' Cost: {cost:.4f}')

    # no need to build the computation graph for backprop when computing accuracy
    model.eval()
    with torch.set_grad_enabled(False):
        train_acc, train_loss = compute_accuracy_and_loss(model, train_data_loader, device=DEVICE)
        valid_acc, valid_loss = compute_accuracy_and_loss(model, val_data_loader, device=DEVICE)
        train_acc_lst.append(train_acc)
        valid_acc_lst.append(valid_acc)
        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
        print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
              f' | Validation Acc.: {valid_acc:.2f}%')
        
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')
  
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')


train_acc_lst = [i.cpu() for i in train_acc_lst]
train_acc_lst = [i.numpy() for i in train_acc_lst]
valid_acc_lst = [i.cpu() for i in valid_acc_lst]
valid_acc_lst = [i.numpy() for i in valid_acc_lst] 

import matplotlib.pyplot as plt

xx = range(NUM_EPOCHS)
plt.plot(xx, train_loss_lst, marker='.', label='train_loss')
plt.plot(xx, valid_loss_lst, marker='.', label='valid_loss')
plt.title('loss record')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.show()

plt.plot(xx, train_acc_lst, marker='.', label='train_acc')
plt.plot(xx, valid_acc_lst, marker='.', label='valid_acc')
plt.title('acc record')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.grid()
plt.show()