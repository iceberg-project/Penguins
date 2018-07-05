from models.models import create_model
from data.png_dataset import PngDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
from options.test_options import TestOptions
import time
from data.data_loader import CreateDataLoader
import torch
import os.path
import argparse
from scipy import misc
from m_util import *
from vis import * 
opt = TestOptions().parse()
opt.model ='single_unet'
opt.checkpoints_dir ='/gpfs/projects/LynchGroup/Penguin_workstation/checkpoints/'
opt.name = 'MSEnc3__bias0.5_bs128_512_768_all'
opt.which_epoch = 25
opt.step = 128
opt.size = 256
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.no_dropout = True
model = create_model(opt)
opt.dataset = '/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/fullsize/'
sdmkdir(opt.dataset+'/output')
file = 'WV03_20160121015319_104001001762AC00_16JAN21015319-M1BS-500638671020_01_P001_u08rf3031.png'
filename = opt.dataset + 'A/' + file
last = time.time()
# your code
im = misc.imread(filename)
elapsed_time = time.time() - last
last = time.time()
print('read im: %0.4f'%(elapsed_time))

w,h,c = im.shape
patches = png2patches(im,opt.step,opt.size)
print(patches.shape)
elapsed_time = time.time() - last
last = time.time()
print('im 2 patches: %0.4f'%(elapsed_time))

orishape = np.asarray(patches.shape)
orishape[-1] = 1

patches = np.reshape(patches, (-1,256,256,3))
outshape  = np.asarray(patches.shape)
outshape[3] = 1
patches = np.transpose(patches,(0,3,1,2))
s = np.asarray(patches.shape)
s[1] = 1
bs = 96
n_patches = patches.shape[0]
out = np.zeros(s) 
print('numbers of patches %d'%(n_patches))
print('Processing all the patches')
for i in range(0,n_patches,bs):
    batch  = patches[i:i+bs,:,:,:]
    batch = torch.from_numpy(batch).float().div(255)
    batch = (batch  - 0.5) * 2
    temp = model.get_prediction_tensor(batch)
    out[i:i+bs,:,:,:] = temp['raw_out']
    
    #print(temp['raw_out'].shape)

elapsed_time = time.time() - last
last = time.time()
print('patches 2 prediction: %0.4f'%(elapsed_time))

print(patches.shape)
print(out.shape)
out = np.reshape(out,outshape)
out = np.reshape(out,(orishape[0],orishape[1],outshape[3],outshape[1],outshape[2]))

print(out.shape)
outpng = patches2png_legacy(out,w,h,opt.step,opt.size)
outpng = np.transpose(outpng,(1,2,0))
outpng = np.squeeze(outpng) 
outpng[outpng<0.5] = 0
outpng[outpng>=0.5] = 1
outpng = outpng*255
misc.toimage(outpng.astype(np.uint8),mode='L').save(opt.dataset+'/output/'+opt.name+'_'+str(+opt.which_epoch)+'/'+file)
