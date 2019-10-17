import torch 
from denseunet import DenseUnet

model = DenseUnet(arch='121')
print(model)
