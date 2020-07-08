import torch
import sys
sys.path.insert(0, "./..")
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
class WeaklyAnnoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.GTroot = opt.dataroot
        self.A_dir = opt.dataroot + '/A/'
        self.B_dir = opt.dataroot + '/B/'
        self.all = []
        self.pos_only = []
        self.strong_only = []
        self.pos_and_strong_only = []
        self.pos_and_weak_only = []
        self.neg_only = []
        self.weak_only = []
        self.neg_and_strong_only = []
        self.neg_and_weak_only = []
        for root,_,fnames in sorted(os.walk(self.A_dir)):
            for fname in fnames:
                if fname.endswith('.png'):
                    X = dict()
                    X['im_path'] = os.path.join(root,fname)
                    X['mask_path'] = os.path.join(self.B_dir,fname)
                    X['ispos'] = True
                    X['imname'] = fname
                    X['isstrong'] = True
                    if not os.path.isfile(X['mask_path']):
                        X['ispos'] = False
                        X['mask_path'] = 'None'
                    if X['ispos']:
                        self.pos_only.append(X)
                        self.pos_and_strong_only.append(X)
                    else:
                        self.neg_only.append(X)
                        self.neg_and_strong_only.append(X)
                    self.all.append(X)
                    self.strong_only.append(X)

                    
        if not hasattr(opt,'wdataroot'):
            opt.wdataroot = '/mnt/train_weakly/PATCHES/256_384/'

        self.wA_dir = opt.wdataroot + '/A/'
        self.wB_dir = opt.wdataroot + '/B/'

        for root,_,fnames in sorted(os.walk(self.wA_dir)):
            for fname in fnames:
                if fname.endswith('.png'):
                    X = dict()
                    X['im_path'] = os.path.join(root,fname)
                    X['mask_path'] = os.path.join(self.wB_dir,fname)
                    X['ispos'] = True
                    X['imname'] = fname
                    X['isstrong'] = False
                    if not os.path.isfile(X['mask_path']):
                        X['ispos'] = False
                        X['mask_path'] = 'None'
                    if X['ispos']:
                        self.pos_only.append(X)
                        self.pos_and_weak_only.append(X)
                    else:
                        self.neg_only.append(X)
                        self.neg_and_weak_only.append(X)
                    self.weak_only.append(X)
                    self.all.append(X)
         
        self.nim = len(self.all)
        self.stats()
    def stats(self):
        print("Dataset type: %s "%(self.name()))
        print("Total Image: %d "%(len(self.all)))
        print("Pos/Neg: %d / %d "%(len(self.pos_only),len(self.all)-len(self.pos_only)))
        print("Accurate Anno / Weak Anno: %d / %d "%(len(self.strong_only),len(self.all)-len(self.strong_only)))
        print("Positive+Accurately Annotated : %d "%(len(self.pos_and_strong_only)))
    def __len__(self):
	    return len(self.strong_only)*2 
    def name(self):
        return 'WeaklyAnnotated DATASET'
    
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
            self.opt.loadSize = np.random.randint(257,400,1)[0]
     
        if not hasattr(self.opt,'s_pos'):
            self.opt.s_pos = 0.5
            self.opt.s_strong = 0.5
        #adaptive sampling:
        if random.random()<self.opt.s_pos:
            if random.random()<self.opt.s_strong:
                choosen_set='pos_and_strong_only'
            else:
                choosen_set='pos_and_weak_only'
        else:
            if random.random()<self.opt.s_strong:
                choosen_set='neg_and_strong_only'
            else:
                choosen_set='neg_and_weak_only'
        r_index = index%(len(getattr(self,choosen_set)))
        data_point = getattr(self,choosen_set)[r_index]       
        #r_index = index % len(self.all)
        #data_point = self.all[r_index]
        A_img = Image.open(data_point['im_path'])
        if data_point['ispos']:
            B_img = Image.open(data_point['mask_path'])
        else:
            t = A_img.size
            B_img = Image.fromarray(np.zeros((A_img.size[0],A_img.size[1])))
        imname = data_point['imname']
        
        
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
        isweak = 0 if data_point['isstrong'] else 1
        return  {'A': A_img, 'B': B_img,'imname':imname,'counts':counts, 'isweak':isweak}
def main():
    import argparse
    opt = argparse.ArgumentParser()
    opt.parse_args() 
    opt.randomSize=True
    opt.keep_ratio=True
    opt.fineSize = 256
    opt.dataroot ='/mnt/train_ori/PATCHES/128_386/'
    A = WeaklyAnnoDataset()
    A.initialize(opt)
    A.stats()
    
    print(A[0])
    
	
if __name__=='__main__':
    main()

