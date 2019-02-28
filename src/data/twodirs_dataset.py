import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from PIL import ImageFilter
import torch
from pdb import set_trace as st
import random
import numpy as np
import time
class TwodirsDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        print self.dir_A
        self.A_paths,self.imname = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        self.B_size = self.A_size
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transformA = transforms.Compose(transform_list)
        self.transformB = transforms.Compose([transforms.ToTensor()])
    def __getitem2__(self, index):
        return getitemxy(self,index,-1,-1)
    def __getitem__(self,index):

        A_path = self.A_paths[index % self.A_size]
        imname = self.imname[index % self.A_size]
        
        index_A = index % self.A_size
        B_path = os.path.join(self.dir_B,imname.replace('.jpg','.png'))
        if not os.path.isfile(B_path):
            B_path = os.path.join(self.dir_B,imname)
        A_img = Image.open(A_path).convert('RGB')
        ow = A_img.size[0]
        oh = A_img.size[1]
        w = np.float(A_img.size[0])
        h = np.float(A_img.size[1])
        if os.path.isfile(B_path): 
            B_img = Image.open(B_path)
        else:
            print B_path
            B_img = Image.fromarray(np.zeros((int(w),int(h)),dtype = np.uint8),mode='L')
        
        if self.opt.randomSize:
            self.opt.loadSize = np.random.randint(257,300,1)[0]
        
        if self.opt.keep_ratio:
            if w>h:
                ratio = np.float(self.opt.loadSize)/np.float(h)
                neww = np.int(w*ratio)
                newh = self.opt.loadSize
            else:
                ratio = np.float(self.opt.loadSize)/np.float(w)
                neww = self.opt.loadSize
                newh = np.int(h*ratio)
        else:
            neww = self.opt.loadSize
            newh = self.opt.loadSize
        
        if self.opt.tsize:
            neww = self.opt.tw
            newh = self.opt.th
        
        A_img = A_img.resize((neww, newh),Image.NEAREST)
        B_img = B_img.resize((neww, newh),Image.NEAREST)
        C_img = []
        
        w = A_img.size[0]
        h = A_img.size[1]
     #   if random.random < 0.2:
     #       mask = B_img.repeat(1,1,3)
     #       A_img[B_img>0] = A_img[B_img>0] * 1.2

        if self.opt.log_scale>0:
            A_img = np.asarray(A_img).astype(np.double) + 1
            A_img = A_img*(float(self.opt.log_scale)/256)
            A_img = np.log(A_img)
            A_img = (A_img)/(np.log(self.opt.log_scale)) *255
        A_img = self.transformA(A_img)
        B_img = self.transformB(B_img)
        if self.opt.useC > 0:
            C_path = os.path.join(self.dir_C,imname.replace('.jpg','.png'))
            C_img = Image.open(C_path) 
            C_img = C_img.resize((neww, newh),Image.NEAREST)
            C_img = np.array(C_img).astype(np.uint8)
            C_img = np.transpose(C_img,(2,0,1))     
            C_img = torch.from_numpy(np.array(C_img)).float() -1
            #C_img = torch.unsqueeze(C_img,0)
            #C_img = self.transform(C_img)

        if not self.opt.no_crop:        
            w_offset = random.randint(0,max(0,w-self.opt.fineSize-1))
            h_offset = random.randint(0,max(0,h-self.opt.fineSize-1))
                
            A_img = A_img[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            B_img = B_img[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]      
            if self.opt.useC >0:
                C_img = C_img[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]      
        
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(2, idx)
            B_img = B_img.index_select(2, idx)
            if self.opt.useC >0:
                C_img = C_img.index_select(2,idx)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path,'imname':imname,'w':ow,'h':oh,'C':C_img}


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'TwodirsDataset'
