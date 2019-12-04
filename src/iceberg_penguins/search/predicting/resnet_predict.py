"""
Wrapper for Segmentation Evaluation. 
Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""
import sys
sys.path.insert(0, "./..")
import torch
import os.path
import argparse
from scipy import misc
from models import create_model
from data.png_dataset import PngDataset
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import CreateDataLoader
from data_processing.m_util import *
#from util.misc import crf_refine 
from data_processing.im_vis import *
import time
import numpy as np
from sklearn.metrics import average_precision_score as ap_score
from sklearn.metrics import jaccard_similarity_score as iou_score
class Pipe:
    def __init__(self,testset,output):
        self.import_model()
        self.output = output +'/' + self.opt.name+'/' + str(self.opt.which_epoch) + '/' +testset +'/'
        self.out_eval = os.path.join(self.output,'eval')
        self.out_raw = os.path.join(self.output,'raw')
        #self.out_crf = os.path.join(self.output,'crf')
        self.out_vis = os.path.join(self.output,'vis')
        sdmkdir(self.output)
        sdmkdir(self.output+'raw')
        sdmkdir(self.output+'vis')
        sdmkdir(self.output+'eval')
        #sdmkdir(self.out_crf)
        self.input =  input
    def import_model(self):
        opt = TestOptions().parse()
        opt.model ='resnet18c'
        opt.checkpoints_dir='/nfs/bigdisk/hieule/checkpoints_CVPR19W/'
        opt.name='resnet18c_bs96_sampling0.5_baseline_resnet'
        opt.which_epoch='400'
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.no_dropout = True
        opt.gpu_ids = [1]
        self.network = create_model(opt)
        self.network.eval()
        self.opt = opt 
    def png_predict(self,im):
        last = time.time()
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opt = parser.parse_args()
        opt.step = 92
        opt.size = 256
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
        bs = 32
        n_patches = patches.shape[0]
        out = np.zeros(s) 
        print('numbers of patches %d'%(n_patches))
        print('Processing all patches')
        for i in range(0,n_patches,bs):
            batch  = patches[i:i+bs,:,:,:]
            batch = torch.from_numpy(batch).float().div(255)
            batch = (batch  - 0.5) * 2
            temp = self.network.get_prediction_tensor(batch)
            out[i:i+bs,:,:,:] = temp['raw_out']
            
            #print(temp['raw_out'].shape)

        elapsed_time = time.time() - last
        last = time.time()
        print('patches 2 prediction: %0.4f'%(elapsed_time))
        out = np.reshape(out,outshape)
        out = np.reshape(out,(orishape[0],orishape[1],outshape[3],outshape[1],outshape[2]))

        outpng = patches2png_legacy(out,w,h,opt.step,opt.size)
        print('merging')
        outpng = np.transpose(outpng,(1,2,0))
        outpng = np.squeeze(outpng) 
        outpng = (outpng + 1)/2
        out = outpng
        outpng = outpng*255
        return outpng
    
    def dir_png_predict(self,fold,outpath):
        self.input = fold
        imlist =[]
        imnamelist =[]
        for root,_,fnames in sorted(os.walk(fold)):
            for fname in fnames:
                if fname.endswith('.png'):
                    path = os.path.join(root,fname)
                    imlist.append((path,fname))
                    imnamelist.append(fname)
        print(len(imnamelist))
        c=1
        pos_pos = 0
        neg_pos = 0
        pos_neg = 0
        neg_neg = 0
        for path,name in imlist :
            inpng = misc.imread(path)
            maskpath = path.replace('/A/','/B/')
            if os.path.isfile(maskpath):
                ispos = True
            else:
                ispos =False

            inpng = misc.imresize(inpng,(256,256))
            inpng = np.reshape(inpng,(-1,256,256,3))
            inpng = np.transpose(inpng,(0,3,1,2))
            inpng =torch.from_numpy(inpng).float().div(255)
            inpng = inpng*2-1
            out = self.network.get_prediction_tensor(inpng)
            out=np.squeeze(out['raw_out'])
            if out>=0:
                if ispos:
                    pos_pos = pos_pos+1
                else:
                    neg_pos = neg_pos +1
            if out<0:
                if ispos:
                    pos_neg = pos_neg+1
                else:
                    neg_neg = neg_neg +1
        print("++/+-/-+/--: %d, %d %d %d"%(pos_pos,pos_neg,neg_pos,neg_neg))

 #           F = open(outpath+'/'+name+'.txt',"w")

 #           F.write("%f"%out)
 #           F.close()
 #           c=c+1
 #           print(c)
        
if __name__=='__main__':

#    a = Pipe('crozier','./Test')
#    a.dir_png_predict('/nfs/bigbox/hieule/GAN/data/Penguins/WL_Train/shannon5_padding400/RP_crozier_guanoarea/A')
 #   a = Pipe('arth','./Test')
 #   a.dir_png_predict('/nfs/bigbox/hieule/GAN/data/Penguins/WL_Train/shannon5_padding400/RP_arthurson_guanoarea/A')
    a = Pipe('testset','./Test')
    a.dir_png_predict('/nfs/bigbox/hieule/GAN/data/Penguins/Test/PATCHES/64_256/A',[])
#    a.dir_png_predict('/nfs/bigbox/hieule/GAN/data/Penguins/WL_Train/merged/PATCHES/192_384/A','/nfs/bigbox/hieule/GAN/data/Penguins/WL_Train/merged/PATCHES/192_384/C')
