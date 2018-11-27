import torch,cv2
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
from options.test_options import TestOptions
import time
import os.path
import argparse
from scipy import misc
from m_util import *
opt = TestOptions().parse()
opt.checkpoints_dir = '/nfs/bigbox/hieule/VU_test/checkpoints/'
opt.which_epoch = 150
opt.input_nc =3
opt.model = 'vusingle_unet'
opt.test = True
opt.no_dropout = True
opt.name = 'VU'
def do_the_thing_please(opt):
    opt.dataroot = opt.im_fold
    opt.res = opt.im_fold + '/res/' + opt.name+ '/'
    sdmkdir(opt.res)
    A_fold = opt.im_fold + 'A/'
    B_fold = opt.im_fold +  'B/'
    imlist=[]
    imnamelist=[]
    
    
    opt.dataset_mode ='png'
    opt.nThreads = 2
    opt.fineSize = 256
    opt.loadSize = 256
    opt.biased_sampling = -0.5
    opt.batchSize = 32
    opt.max_dataset_size = 5000000
    opt.no_flip = True
    opt.serial_batches = False

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(len(dataset))
    model = create_model(opt)
    print(model)
    for i,data in enumerate(dataset):
        print data['A'].shape
        temp = model.get_prediction(data)['raw_out']
        print temp.shape
        for k in range(0,temp.shape[0]):
            misc.toimage(temp[k,:,:,0],mode='L').save(os.path.join(opt.res,data['imname'][k]))

opt.name = 'VU'
opt.im_fold = '/nfs/bigmind/vhnguyen/data/ivygap/train/'
do_the_thing_please(opt)

#opt.im_fold ='/nfs/bigbox/hieule/penguin_data/CROPPED/p500_train/'
#do_the_thing_please(opt)
#opt.im_fold_temp ='/nfs/bigbox/hieule/penguin_data/Test/*TEST*/CROPPED/p300/'
#for t in ["PAUL","CROZ"]:
#    opt.im_fold = opt.im_fold_temp.replace("*TEST*",t)
#    do_the_thing_please(opt)
