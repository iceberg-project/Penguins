import torch
from torch import nn as nn
import torchvision
from torchvision import datasets, models, transforms

class resnet18(nn.Module):
    def __init__(self,out_nc=1, gpu_ids=[]):
        super(resnet18,self).__init__()
        self.gpu_ids = gpu_ids
        model_ft = models.resnet18(pretrained=True) 
        num_ftrs = model_ft.fc.in_features  
        model_ft.fc = nn.Linear(num_ftrs, out_nc) 
        self.model=model_ft
        if len(gpu_ids) > 0:
            self.model.cuda(gpu_ids[0])

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

def main():
    a = resnet18(1)
    b= torch.rand(10,3,224,224)
    k = a(b)
    print(k.shape)

if __name__=="__main__":
    main()
