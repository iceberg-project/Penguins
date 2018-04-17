
import argparse
from data.mb_dataset import MBDataset
import time
from data.data_loader import CreateDataLoader
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opt = parser.parse_args()
opt.dataset_mode ='mb'
opt.nThreads = 2
opt.fineSize = 300
opt.loadSize = 380
opt.biased_sampling = 0.5
opt.batchSize = 64
opt.max_dataset_size = 10000
opt.serial_batches = False
opt.dataroot = '/nfs/bigbox/hieule/penguin_data/MB_Same_Size/Train/Train_all/CROPPED/p500_train/PATCHES/64_386/'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
for i,data in enumerate(dataset):
    print data['imname']
