"""
Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""
import torch
from torch import nn as nn
import torchvision
from torchvision import datasets, models, transforms

class resnet18(nn.Module):
    def __init__(self,out_nc=1, gpu_ids=[]):
        super(resnet18,self).__init__()
        self.gpu_ids = gpu_ids
        net = models.resnet18(pretrained=True) 
        features=list()
        features = list(net.children())[:-1] # Remove last layer
        features.extend([nn.Conv2d(512, out_nc, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), bias=True)])
        self.model = nn.Sequential(*features) # Replace the model classifier$
        #num_ftrs = net.fc.in_features  
        #net.fc = nn.Linear(num_ftrs, out_nc) 
        #self.model=net[:-2]
        if len(gpu_ids) > 0:
            self.model.cuda(gpu_ids[0])
        print(self.model)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

def main():
    a = resnet18(1)
    b= torch.rand(10,3,224,224)
    b = torch.rand(10,3,300,300)
    k = a(b)
    print(k.shape)

if __name__=="__main__":
    main()
