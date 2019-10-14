import torch 
from models.dense-unet import DenseUnet

model = DenseUnet(arch='201')
print(model)
