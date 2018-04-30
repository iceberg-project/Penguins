from shutil import copyfile
import argparse
import os.path
from m_util import *
opt = argparse.ArgumentParser().parse_args()
#opt.im_fold = '/nfs/bigbox/hieule/penguin_data/CROPPED/p500'
opt.im_fold = '/nfs/bigbox/hieule/penguin_data/p1000'
opt.split = '/nfs/bigbox/hieule/penguin_data/p1000/split'
sdmkdir(opt.split)
def choose_n_k(n,k):
    idx = np.random.permutation(n)
    return idx[0:k],idx[k:]

imlist = []
test_ratio = 0.2

for root,_,fnames in sorted(os.walk(opt.im_fold+'/A')):
    for fname in fnames:
        if fname.endswith('.png') :
            imlist.append(fname)
nim = len(imlist)
print "all: %d"%(nim)
print imlist
nim_test = int(float(nim)*test_ratio)
print nim_test
nim_train = nim - nim_test

for split_idx in [0]:
    test_list= []
    train_list = []
    testidx,trainidx  = choose_n_k(nim,nim_test)
    print trainidx
    test_list = [imlist[i] for i in testidx]
#    test_list = [
#    'orthoWV02_11FEB191312281-M1BS-10300100098AAB00_u08rfAEAC.png',
#    'WV02_20160218213346_1030010053BC0300_16FEB18213346-M1BS-500638407020_01_P002_u08rf3031.png',
#    'WV02_20110131195115_1030010009CCF900_11JAN31195115-M1BS-052549143040_01_P003_u08rf3031.png',
 #   'WV03_20170217064537_10400100297FEA00_17FEB17064537-M1BS-057107305010_01_P001_u08rf3031.png'
    ]
    print "test:"
    print test_list
    train_list = [i for i in imlist if  i not in test_list]
    print "train:"
    print train_list

    file = open(opt.split+"/test_"+str(split_idx),"w")
    for i in test_list:
        file.write(i[:-4]+"$")
    file.close()



    file = open(opt.split+"/train_"+str(split_idx)+"_all","w")
    for i in train_list:
        file.write(i[:-4]+"$")
    file.close()

    print nim_train
    print train_list
    print len(train_list)

    ratioa = [0.25,0.5,0.75]
    for ratio in ratioa:
        ntrain = int(float(nim_train)*ratio) 
        for iter in range(0,5):
            print nim_train,ntrain
            tidx,_=choose_n_k(nim_train,ntrain)
            print tidx
            tlist = [train_list[i] for i in tidx]
            file = open(opt.split+"/train_"+str(split_idx)+'_'+str(ratio)+'_'+str(iter),"w")
            for i in tlist:
                file.write(i[:-4]+"$")
            file.close()

            
