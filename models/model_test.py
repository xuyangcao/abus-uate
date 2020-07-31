import torch 
from denseunet import DenseUnet
from resunet import UNet 

model = DenseUnet(arch='121')
print(model)
