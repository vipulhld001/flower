import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNetV2

# Note the model and functions here defined do not have any FL-specific components.
from blocks import BaseBlock
import math

class Net(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    # def __init__(self, num_classes: int) -> None:
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 4 * 4, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, num_classes)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 4 * 4)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    #FOR CIFAR 10 WOrking one


    # def __init__(self, num_classes: int) -> None:
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, num_classes)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     #x = torch.flatten(x, 1)
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    
    #FOR MOBILENETV2
    # def __init__(self, num_classes: int, alpha = 1):
    #     super(Net, self).__init__()
    #     self.num_classes = num_classes

    #     # first conv layer 
    #     self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
    #     self.bn0 = nn.BatchNorm2d(int(32*alpha))

    #     # build bottlenecks
    #     BaseBlock.alpha = alpha
    #     self.bottlenecks = nn.Sequential(
    #         BaseBlock(32, 16, t = 1, downsample = False),
    #         BaseBlock(16, 24, downsample = False),
    #         BaseBlock(24, 24),
    #         BaseBlock(24, 32, downsample = False),
    #         BaseBlock(32, 32),
    #         BaseBlock(32, 32),
    #         BaseBlock(32, 64, downsample = True),
    #         BaseBlock(64, 64),
    #         BaseBlock(64, 64),
    #         BaseBlock(64, 64),
    #         BaseBlock(64, 96, downsample = False),
    #         BaseBlock(96, 96),
    #         BaseBlock(96, 96),
    #         BaseBlock(96, 160, downsample = True),
    #         BaseBlock(160, 160),
    #         BaseBlock(160, 160),
    #         BaseBlock(160, 320, downsample = False))

    #     # last conv layers and fc layer
    #     self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
    #     self.bn1 = nn.BatchNorm2d(1280)
    #     self.fc = nn.Linear(1280, num_classes)

    #     # weights init
    #     self.weights_init()


    # def weights_init(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))

    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


    # def forward(self, inputs):

    #     # first conv layer
    #     x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
    #     # assert x.shape[1:] == torch.Size([32, 32, 32])

    #     # bottlenecks
    #     x = self.bottlenecks(x)
    #     # assert x.shape[1:] == torch.Size([320, 8, 8])

    #     # last conv layer
    #     x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
    #     # assert x.shape[1:] == torch.Size([1280,8,8])

    #     # global pooling and fc (in place of conv 1x1 in paper)
    #     x = F.adaptive_avg_pool2d(x, 1)
    #     x = x.view(x.shape[0], -1)
    #     x = self.fc(x)

    #     return x

    #LENET 5

    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    




def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy