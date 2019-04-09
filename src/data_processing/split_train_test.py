"""
Split dataset to testing/training set
Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""
from shutil import copyfile
import argparse
import os.path
from m_util import *
opt = argparse.ArgumentParser().parse_args()
#opt.im_fold = '/nfs/bigbox/hieule/penguin_data/CROPPED/p500'
opt.im_fold='/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/p1000/'
opt.split = opt.im_fold+'split2/'
sdmkdir(opt.split)
def gen_folds(NIM,n):
    idx = np.random.permutation(NIM)
    return [idx[i::n] for i in range(n)]
def choose_n_k(n,k):
    idx = np.random.permutation(n)
    return idx[0:k],idx[k:]

imlist = []
test_ratio = 0.2

for root,_,fnames in sorted(os.walk(opt.im_fold+'/A')):
    for fname in fnames:
        if fname.endswith('.png') and 'M1BS' in fname:
            imlist.append(fname)
nim = len(imlist)
print("all: ",nim)
print(imlist)
list_to_file(opt.split+'/all.txt',imlist)
nim_test = int(float(nim)*test_ratio)
print(nim_test)
nim_train = nim - nim_test
n_folds=5
for split_idx in range(5):
    test_list= []
    train_list = []
    folds = gen_folds(nim,n_folds)
    test_list = [imlist[i] for i in folds[0]] 
    print("test len:", len(test_list))
    train_list = [i for i in imlist if  i not in test_list]
    list_to_file(opt.split+"/test_"+str(split_idx)+'.txt',test_list)
    for fold_i in range(1,n_folds):
        trainidx= []
        for i in range(1,fold_i+1):
            for item in folds[i]:
                trainidx.append(item)
        tlist = [imlist[i] for i in trainidx]
        print("train fold %d len:", fold_i,len(trainidx))
        file_n = opt.split+"/train_"+str(split_idx)+'_'+str(fold_i)+'.txt'
        list_to_file(file_n,tlist)

            
