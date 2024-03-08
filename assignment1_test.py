import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define data transforms
transform = transforms.Compose([
    transforms.RandomCrop(256, padding=32, padding_mode='reflect'), 
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize
])

# Define paths to your data directories
test_data_dir = './Birds_25/test'

# Create datasets
test_dataset = ImageFolder(root=test_data_dir, transform=transform)

# Create data loaders
batch_size = 32  # Reduced batch size for better memory utilization
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Accessing labels:
# print("Labels:", train_dataset.classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        identity = x
            
        out = self.relu(self.batch1(self.conv1(x)))
        out = self.batch2(self.conv2(out))
        if not self.downsample:
            out += identity
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=25):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1])
        self.layer3 = self._make_layer(block, 64, layers[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=2,padding=1),
                nn.BatchNorm2d(planes)
            )

        layers = []
        self.inplanes = planes
        layers.append(block(self.inplanes, planes,1, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        return x
    
num_layers = 2

model = ResNet(ResBlock, [num_layers, num_layers, num_layers]).to(device)
model.load_state_dict(torch.load('my_model.pth'))

classes =  ['Asian-Green-Bee-Eater', 'Brown-Headed-Barbet', 'Cattle-Egret', 'Common-Kingfisher', 'Common-Myna', 'Common-Rosefinch', 'Common-Tailorbird', 'Coppersmith-Barbet', 'Forest-Wagtail', 'Gray-Wagtail', 'Hoopoe', 'House-Crow', 'Indian-Grey-Hornbill', 'Indian-Peacock', 'Indian-Pitta', 'Indian-Roller', 'Jungle-Babbler', 'Northern-Lapwing', 'Red-Wattled-Lapwing', 'Ruddy-Shelduck', 'Rufous-Treepie', 'Sarus-Crane', 'White-Breasted-Kingfisher', 'White-Breasted-Waterhen', 'White-Wagtail']

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(25)]
    n_class_samples = [0 for _ in range(25)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
        del images,labels,outputs

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network for val: {acc} %')

    for i in range(25):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of for val {classes[i]}: {acc} %')