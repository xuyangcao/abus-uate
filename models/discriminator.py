import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 
		return x

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
        out = self.sigmoid(self.fc(out))
        #print('out.shape: ', out.shape) # bx1
        
        return out, conv4_maps
