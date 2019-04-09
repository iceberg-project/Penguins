import sys
sys.path.insert(0, "./..")
from data.png_dataset import PngDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
from options.test_options import TestOptions
import time
import torch
import os.path
import argparse
from scipy import misc
#from vis import visAB,visABC
from m_util import *
parse = argparse.ArgumentParser()
parse.add_argument('--dataset')
opt = parse.parse_args()
#opt.root = '/mnt/train_weakly/'
opt.root = '/nfs/bigbox/hieule/GAN/data/Penguins/Test/'
#opt.root = '/nfs/bigbox/hieule/GAN/data/Penguins/WL_Train/merged/'
#opt.im_fold_temp ='/nfs/bigbox/hieule/penguin_data/Test/*TEST*/CROPPED/p300/'
#for t in ["PAUL","CROZ"]:
#opt.im_fold = opt.im_fold_temp.replace("*TEST*",t)
opt.im_fold = opt.root
opt.step = 64 #128 for testing, 64 for training
opt.size = 256 #256 for testing, 386 for training
opt.patch_fold_A = opt.im_fold+'PATCHES/'+str(opt.step)+'_'+ str(opt.size)+ '/A/'
opt.patch_fold_B = opt.im_fold+'PATCHES/'+str(opt.step)+'_'+ str(opt.size)+'/B/'
A_fold = opt.im_fold + 'A/'
B_fold = opt.im_fold +  'B/'

opt.input_nc =3
sdmkdir(opt.patch_fold_A)
sdmkdir(opt.patch_fold_B)
imlist=[]
todolist=[]
#todolist = read_list('/nfs/bigbox/hieule/penguin_data/p1000/split/test_new')
print(todolist)
imnamelist=[]

for root,_,fnames in sorted(os.walk(A_fold)):
    for fname in fnames:
        if fname.endswith('.png') and 'M1BS' in fname and not '_._' in fname :
            path = os.path.join(root,fname)
            path_mask = os.path.join(B_fold,fname)
            imlist.append((path,path_mask,fname))
            imnamelist.append(fname)
c =0 
for im_path,mask_path,imname in  imlist:
    c= c+1
    print(c)
    try:
        png = misc.imread(im_path,mode='RGB')
        mask = misc.imread(mask_path)
        if mask.shape[0]<opt.size or mask.shape[1]<opt.size:
                mask = misc.imresize(mask,size=(opt.size,opt.size),mode='L')
                png = misc.imresize(png,size=(opt.size,opt.size))
                    
        w,h,z = png.shape
        savepatch_train(png,mask,w,h,opt.step,opt.size,opt.patch_fold_A+'/'+imname[:-4]+'#',opt.patch_fold_B+'/'+imname[:-4]+'#')
    except:
        print('cant read image: %s'%imname)
        continue
