import torch
import torch.nn as nn
import numpy as np
from dataset import Dataset50Loader
import cv2
import torch.nn.utils.prune as prune
class LeNet5(nn.Module):
    def __init__(self) :
        super(LeNet5,self).__init__()
        self.allnn=nn.Sequential(
            nn.Conv2d(3,6,5),   #64->60
            nn.Sigmoid(),
            nn.MaxPool2d(2,stride=2),#60->30
            nn.Conv2d(6,16,5),          #30->26
            nn.Sigmoid(),
            nn.MaxPool2d(2,stride=2),   #26->13
            nn.Conv2d(16,120,5), #13->9
        )
        self.fn=nn.Sequential(
            nn.Linear(120*9*9,84),
            nn.Sigmoid(),
            nn.Linear(84,50),
            nn.Softmax(dim=1),
        )
    def forward(self,x):

        x=self.allnn(x)
        x=x.view(-1,120*9*9)
        x=self.fn(x)

        return x
    
device = torch.device("cpu")    
model=LeNet5().to(device)
model.load_state_dict(torch.load("tor_model_49.pth",map_location=device))
####################### prunING

for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight',amount=0.6)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.6)

torch.save(model, 'quantized_model_1.pth')


########################## PTQ
# model_int8 = torch.ao.quantization.quantize_dynamic(
#     model,  # the original model
#     {torch.nn.Linear,torch.nn.Conv2d,torch.nn.Sigmoid},  # a set of layers to dynamically quantize
#     dtype=torch.qint8)
#torch.save(model_int8.state_dict(), 'quantized_model.pth')



