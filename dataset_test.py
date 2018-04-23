import argparse
import time
from data.data_loader import CreateDataLoader
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opt = parser.parse_args()
opt.dataset_mode ='png_withlist'
opt.istrain = False
opt.nThreads = 2
opt.fineSize = 300
opt.loadSize = 380
opt.biased_sampling = 0.5
opt.batchSize = 64
opt.max_dataset_size = 1000000
opt.serial_batches = False
opt.traininglist = '/nfs/bigbox/hieule/penguin_data/p1000/train1_all'
opt.dataroot = '/nfs/bigbox/hieule/penguin_data/p1000/PATCHES/64_386/'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
print len(dataset)
for i,data in enumerate(dataset):
    print data['imname'][0]

