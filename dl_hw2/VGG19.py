import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()
        # Consists of a sequence of convolutional and pooling layers
        self.features = self.make_layers([64, 64, 'M', 
                                          128, 128, 'M',
                                          256, 256, 256, 256, 'M', 
                                          512, 512, 512, 512, 'M', 
                                          512, 512, 512, 512, 'M'])
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # A sequence of fully connected (dense) layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            # Add a max pooling layer
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
def vggnet():
    num_classes = 100
    model = VGG19(num_classes)
    return model
