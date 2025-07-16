import torch.nn as nn
import torch.nn.functional as F
import torch

class GAN_Generator(nn.Module):
    def __init__(self, dim_z, dim_c):
        super(GAN_Generator, self).__init__()

        self.dim_z = dim_z
        self.dim_c = dim_c
        self.conditionExpand=nn.Sequential(
            nn.Linear(24, dim_c),
            nn.ReLU(0.2)
        )
        layers = [
            (dim_z + dim_c, 512, 4, 1, 0), 
            (512, 256, 4, 2, 1),
            (256, 128, 4, 2, 1),
            (128, 64, 4, 2, 1),
            (64, 3, 4, 2, 1)
        ]
        net_layers = []
        for i, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(layers):
            net_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            if i < len(layers) - 1:  # Add BatchNorm and ReLU to all but the last layer
                net_layers.append(nn.BatchNorm2d(out_channels))
                net_layers.append(nn.ReLU(True))
        net_layers.append(nn.Tanh())  # Add Tanh at the end

        self.net = nn.Sequential(*net_layers)

    def forward(self, x, label):
        x = x.view(-1, self.dim_z, 1, 1)
        label = self.conditionExpand(label).view(-1, self.dim_c, 1, 1)
        out = torch.cat((x, label), 1)
        return self.net(out)
    
    def initialize_weights(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(0, 0.02)
                self._modules[m].bias.data.zero_()
    

class GAN_Discriminator(nn.Module):
    def __init__(self, num_classes, image_size):
        super(GAN_Discriminator, self).__init__()
        self.image_size = image_size
        self.conditionExpand=nn.Sequential(
            nn.Linear(num_classes, image_size * image_size),
            nn.LeakyReLU(0.2, True)
        )

        layers = [
            (4, 64, 4, 2, 1),        # (3+1) x 64 x 64
            (64, 128, 4, 2, 1),      # (64) x 32 x 32
            (128, 256, 4, 2, 1),     # (128) x 16 x 16
            (256, 512, 4, 2, 1),     # (256) x 8 x 8
            (512, 1, 4, 1, 0)        # (512) x 4 x 4
        ]
        
        net_layers = []
        for i, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(layers):
            net_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            if i == 0:
                net_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if i < len(layers) - 1 and i != 0:  # Add BatchNorm and LeakyReLU to all but the last layer
                net_layers.append(nn.BatchNorm2d(out_channels))
                net_layers.append(nn.LeakyReLU(0.2, inplace=True))
        #net_layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_layers)


    def forward(self, x, label):
        label = self.conditionExpand(label).view(-1, 1, 64, 64)
        out = torch.cat((x, label), 1)
        return self.net(out).view(-1, 1)

    def initialize_weights(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(0, 0.02)
                self._modules[m].bias.data.zero_()
