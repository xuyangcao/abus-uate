import torch 
from denseunet import DenseUnet
from resunet import UNet 

#model = DenseUnet(arch='201')
model = UNet(3, 2, relu=False)
#print(model)
n_params = sum([p.data.nelement() for p in model.parameters()])
print(n_params)
