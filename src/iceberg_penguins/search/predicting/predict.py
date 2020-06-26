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
from ..models import create_model
from ..data.png_dataset import PngDataset
from ..options.train_options import TrainOptions
from ..options.test_options import TestOptions
from ..data import CreateDataLoader
from ..data_processing.m_im_util import *
#from util.misc import crf_refine 
from ..data_processing.im_vis import *
import time
import numpy as np
from sklearn.metrics import average_precision_score as ap_score
from sklearn.metrics import jaccard_similarity_score as iou_score
import imageio

class Pipe:
    def __init__(self,opt):
        self.opt = opt
        self.opt.step = 128
        self.opt.size = 256
        self.m_name = opt.name
        self.epoch = opt.epoch
        self.import_model()
        self.output = opt.output +'/' + self.m_name+'/' + str(self.opt.which_epoch) + '/' +opt.testset +'/'
        #self.out_eval = os.path.join(self.output,'eval')
        self.out_raw = os.path.join(self.output,'raw')
        self.out_vis = os.path.join(self.output,'vis')
        sdmkdir(self.output)
        sdmkdir(self.output+'raw')
        sdmkdir(self.output+'vis')
        sdmkdir(self.output+'tmp')
        self.input =  input
    def import_model(self):
        opt = self.opt
        opt.name = self.m_name
        if 'unetr' in opt.name:
            opt.model='unetr'
        elif 'unet' in opt.name:
            opt.model = 'unet'
        opt.which_epoch=self.epoch
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.no_dropout = True
        opt.gpu_ids = [0]
        self.network = create_model(opt)
        self.network.eval()
        self.opt = opt 
    def list_tif_predict(self,file):
        import rasterio
        from data_processing.tif_handle import TIF_H
        root = '/gpfs/projects/LynchGroup/Orthoed/'
        imlist =[]
        imnamelist =[]
        f = open(file,'r')
        while True:
            line = f.readline()
            if not line:break
            imnamelist.append(line.split()[0] ) 
        print(imnamelist)

        for name in imnamelist :
            self.tif_predict(root+name)
    def dir_tif_predict(self,fold):
        imlist =[]
        imnamelist =[]
        for root,_,fnames in sorted(os.walk(fold)):
            for fname in fnames:
                if fname.endswith('.tif') and 'M1BS' in fname:
                    path = os.path.join(root,fname)
                    imlist.append(path)
                    imnamelist.append(fname)
        print(imnamelist)
        for name in imlist :
            try:
                self.tif_predict(name)
            except:
                print('failed')
    def tif_predict(self,file):
        try:    
            print(file)
            basename = os.path.basename(file)
            if not os.path.isfile(self.output+'/'+basename):
                tif = TIF_H(file)
                tif.get_png()
                outpng = self.png_predict(tif.png)
                print(outpng) 
                tif.profile.update(dtype=rasterio.uint8, count=1)
                with rasterio.open(self.output+'/'+basename, 'w', **tif.profile) as dst:
                        dst.write(outpng.astype(rasterio.uint8), 1)
        except:
            print("failed")
    def png_predict(self,im):
        last = time.time()
        opt.step = self.opt.step#128
        opt.size = self.opt.size#256
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
    
    def dir_png_predict(self,fold):
        self.input = fold
        imlist =[]
        imnamelist =[]
        for root,_,fnames in sorted(os.walk(fold)):
            for fname in fnames:
                if fname.endswith('.png') and 'M1BS' in fname and not fname.startswith('.'):
                    path = os.path.join(root,fname)
                    imlist.append((path,fname))
                    imnamelist.append(fname)
        print(imnamelist)
        for path,name in imlist :
            print(path)
            inpng = misc.imread(path)
            outpng = self.png_predict(inpng)
            outim = show_plainmask_on_image(inpng,outpng)
            misc.imsave(self.output+'/raw/'+name,outpng)
            misc.imsave(self.output+'/vis/'+name,outim)
    def eval_dir_J(self,GT):
        imlist =[]
        imnamelist =[]
        for root,_,fnames in sorted(os.walk(self.out_raw)):
            for fname in fnames:
                if fname.endswith('.png') and 'M1BS' in fname and not fname.startswith('.'):
                    path = os.path.join(root,fname)
                    imlist.append((path,fname))
                    imnamelist.append(fname)
        with open(os.path.join(self.out_raw+'/iou.txt'),'w') as FILE:
            iou_all =[]
            for path, name in imlist:
                preds = misc.imread(os.path.join(self.out_raw,name))
#                preds = (preds>150)
                preds = (preds>50)
                labs = misc.imread(os.path.join(GT,name))
                labs = (labs >0.5)
                misc.imsave(self.output+'/tmp/'+name,preds.astype(np.uint8)*255)
                target= labs.flatten()
                prediction = preds.flatten()
                intersection = np.logical_and(target, prediction)
                union = np.logical_or(target, prediction)
                iou = np.sum(intersection.astype(np.float)) / np.sum(union.astype(np.float))
                print(iou)
                iou_all.append(iou)
                FILE.write(' %02.2f   %s\n'%(iou,name))
            m_iou = sum(iou_all)/len(iou_all)
            print("Mean IOU: %f"%m_iou)
            FILE.write(' %02.2f  \n'%(m_iou))
            FILE.close()

    def eval_dir_AP(self,Outpath,GT):
        imlist =[]
        imnamelist =[]
        for root,_,fnames in sorted(os.walk(GT)):
            for fname in fnames:
                if fname.endswith('.png') and 'M1BS' in fname and not fname.startswith('.'):
                    path = os.path.join(root,fname)
                    imlist.append((path,fname))
                    imnamelist.append(fname)
        with open(os.path.join(Outpath+'/Prec_recall.txt'),'w') as FILE:
            FILE.write('  Prec - Recall  - NAME\n')
            for path, name in imlist:
                preds = misc.imread(os.path.join(Outpath,name))
                labs = misc.imread(os.path.join(GT,name))
                labs = (labs == np.amax(labs)).astype(np.float)
                
                preds = preds.astype(np.uint8)
                AP = ap_score(labs.flatten(),preds.flatten())
                preds = (preds>=(50)).astype(np.float)

                
                tp = np.sum(preds[labs==1] == 1).astype(np.float)
                fn = np.sum(preds[labs==1] == 0).astype(np.float)
                fp = np.sum(preds[labs==0] == 1).astype(np.float)
                tn = np.sum(preds[labs==0] ==0).astype(np.float)

                print(tp,fn,fp,tn)
                conf_matrix = [                        tp/(tp+fn),fp/(tp+fp),fn/(fn+tn),tn/(tn+fn)]
                Prec = tp/(tp+fp)
                Recall = tp/(tp+fn)
            
                FILE.write(' %02.2f  |  %02.2f  |   %02.2f |  %s \n'%(AP,Prec,Recall,name))
            FILE.close()

    def testset1(self):
        self.dir_png_predict('/nfs/bigbox/hieule/GAN/data/Penguins/Test/A')
#        self.eval_dir_J('/nfs/bigbox/hieule/GAN/data/Penguins/Test/B')
    def test_single_png(self,impath):
        name = os.path.basename(impath)
        #inpng = misc.imread(impath)
        #inpng = misc.imread(impath)
        inpng = imageio.imread(impath)
        outpng = self.png_predict(inpng)
        outim = show_plainmask_on_image(inpng,outpng)
        imageio.imsave(self.output+'/raw/'+name,outpng)
        imageio.imsave(self.output+'/vis/'+name,outim)
if __name__=='__main__':
    opt = TestOptions().parse()
    a = Pipe(opt)
    a.test_single_png(opt.input_im)
    #a.testset1()
