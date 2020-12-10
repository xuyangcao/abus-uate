import torch.nn as nn
import torch.nn.functional as F

from models.resunet import * 

class s4GAN_discriminator(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, ndf = 64):
        super(s4GAN_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1) # 160 x 160
        self.conv2 = nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 80 x 80
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 20 x 20
        self.avgpool = nn.AvgPool2d((8, 32))
        self.fc = nn.Linear(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        #print('maps.shape: ', maps.shape) # bx512x1x1
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        #out = self.fc(out)
        out = self.sigmoid(self.fc(out))
        #print('out.shape: ', out.shape) # bx1
        
        return out, conv4_maps

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, relu=True, dropout=0.3):
        super(Discriminator, self).__init__()

        self.input_tr = InputTransition(in_channels, 32, relu)
        self.down_tr64 = DownTransition(32, 5, relu, dropout)
        self.down_tr128 = DownTransition(64, 5, relu, dropout)
        self.down_tr256 = DownTransition(128, 5, relu, dropout)
        self.down_tr512 = DownTransition(256, 5, relu, dropout)

        self.avgpool = nn.AvgPool2d((8, 32))
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_tr(x)
        x = self.down_tr64(x)
        x = self.down_tr128(x)
        x = self.down_tr256(x)
        x = self.down_tr512(x)

        maps = self.avgpool(x)
        conv_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))

        return out, conv_maps
