import torch
import os.path
from scipy import misc
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from PIL import ImageFilter
from pdb import set_trace as st
import random
import numpy as np
import time
class PngDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.GTroot = opt.dataroot
        self.A_dir = opt.dataroot + '/A/'
        self.B_dir = opt.dataroot + '/B/'
        self.imname = []
        self.imname_pos = []
        for root,_,fnames in sorted(os.walk(self.A_dir)):
            for fname in fnames:
                if fname.endswith('.png'):
                    path = os.path.join(root,fname)
                    self.imname.append(fname)
        
        for root,_,fnames in sorted(os.walk(self.B_dir)):
            for fname in fnames:
                if fname.endswith('.png'):
                    path = os.path.join(root,fname)
                    self.imname_pos.append(fname)
        self.nim = len(self.imname)

    def __len__(self):
        return 5000
	#return self.nim
    def name(self):
        return 'PNGDATASET'
    
    def getpatch(self,idx,i,j):
        A_img = self.tifimg[:,i*256:(i+1)*256,j*256:(j+1)*256]
        B_img = self.GTmask[:,i*256:(i+1)*256,j*256:(j+1)*256]
        A_img = torch.from_numpy(A_img).float().div(255)
        B_img = torch.from_numpy(B_img).float().div(255)
        
        A_img = torch.unsqueeze(A_img,0)
        B_img = torch.unsqueeze(B_img,0)
        return  {'A': A_img, 'B': B_img,'imname':self.imname[0]}
    def get_number_of_patches(self,idx):
        return self.nx,self.ny
    def __getitem__(self,index):
        if self.opt.randomSize:
            self.opt.loadSize = np.random.randint(257,300,1)[0]
        if random.random() < self.opt.biased_sampling:
            r_index = index % len(self.imname_pos)
            imname = self.imname_pos[r_index]
            A_img = Image.open(os.path.join(self.A_dir,imname))
            B_img = Image.open(os.path.join(self.B_dir,imname))
        else:
            
            r_index = index % len(self.imname)
            imname = self.imname[r_index]
            A_img = Image.open(os.path.join(self.A_dir,imname))
            
            if imname in self.imname_pos:
                B_img = Image.open(os.path.join(self.B_dir,imname))
            else:
                t = A_img.size
                B_img = Image.fromarray(np.zeros((A_img.size[0],A_img.size[1])))
        ow = A_img.size[0]
        oh = A_img.size[1]
        w = np.float(A_img.size[0])
        h = np.float(A_img.size[1])
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
        t =[Image.FLIP_LEFT_RIGHT,Image.ROTATE_90]
        for i in range(0,2):
            c = np.random.randint(0,3,1,dtype=np.int)[0]
            if c==2: continue
            A_img=A_img.transpose(t[c])
            B_img=B_img.transpose(t[c])
        
        degree=np.random.randint(-10,10,1)[0]
        A_img=A_img.rotate(degree)
        B_img=B_img.rotate(degree)
        
        A_img = A_img.resize((neww, newh),Image.NEAREST)
        B_img = B_img.resize((neww, newh),Image.NEAREST)
        
        A_img = np.asarray(A_img)
        B_img = np.asarray(B_img)
        A_img = A_img[:,:,0:3]

        B_img.setflags(write=1)
        B_img[B_img==2] = 255
        B_img[B_img!=255] = 0
        A_img = np.transpose(A_img,(2,0,1))
        B_img = np.expand_dims(B_img, axis=0)
        z,w,h = A_img.shape
        w_offset = random.randint(0,max(0,w-self.opt.fineSize-1))
        h_offset = random.randint(0,max(0,h-self.opt.fineSize-1))
        A_img = A_img[:, w_offset:w_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize] 
        B_img = B_img[:,w_offset:w_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize]
        A_img = torch.from_numpy(A_img).float().div(255)
        B_img = torch.from_numpy(B_img).float().div(255)
        A_img = A_img - 0.5
        A_img = A_img * 2

        counts = torch.mean(B_img.view(-1,1))
        B_img = B_img - 0.5
        B_img = B_img * 2
        count_ids = 1
        return  {'A': A_img, 'B': B_img,'imname':imname,'counts':counts, 'count_ids':count_ids}
