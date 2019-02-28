import sys
sys.path.insert(0, "./..")
import torch
import os.path
import argparse
from scipy import misc
from models.models import create_model
from data.png_dataset import PngDataset
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from data_processing.tif_handle import TIF_H
from data_processing.m_util import *
#from util.misc import crf_refine 
from data_processing.im_vis import *
import time
import numpy as np
import rasterio
from sklearn.metrics import average_precision_score as ap_score
class Pipe:
    def __init__(self,input,output):
        self.import_model()
        self.output = output + self.opt.name + str(self.opt.which_epoch) + '/'
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
        opt.model ='single_unet'
        opt.checkpoints_dir ='/nfs/bigbox/hieule/penguin_data/checkpoints/'
        opt.name='MSEnc3_p2000_train_bias0.5_bs128'
        opt.which_epoch = 200
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.no_dropout = True
        opt.gpu_ids = [2]
        self.network = create_model(opt)
        self.opt = opt 
    def list_tif_predict(self,file):
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
#    
#    def dir_crf_smoothing(self):
#        imlist =[]
#        imnamelist =[]
#        for root,_,fnames in sorted(os.walk(self.input)):
#            for fname in fnames:
#                if fname.endswith('.png') and 'M1BS' in fname and not fname.startswith('.'):
#                    path = os.path.join(root,fname)
#                    imlist.append((path,fname))
#                    imnamelist.append(fname)
#        print(imnamelist)
#        for path,name in imlist :
#            print(path)
#            im = misc.imread(path)
#            raw_pred = misc.imread(os.path.join(self.out_raw,name))
#            crf_out = crf_refine(im.astype(np.uint8),raw_pred.astype(np.uint8))
#            misc.imsave(self.output+'/crf/'+name,crf_out)


    def eval_dir(self,Outpath,GT):
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



        
if __name__=='__main__':

    a = Pipe('','./test_PTS/')
    a.input ='/nfs/bigbox/hieule/penguin_data/TEST_PTS_MASK/A' 
    im = a.png_predict(misc.imread('/nfs/bigbox/hieule/penguin_data/TEST_PTS_MASK/A/WV02_20151204195602_103001004F9A8500_15DEC04195602-M1BS-500637515080_01_P006_u08rf3031_fixed.png'))
    misc.imsave('chec.png',im)
