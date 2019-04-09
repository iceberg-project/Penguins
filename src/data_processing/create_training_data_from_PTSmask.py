"""
Extract the annotations from PTS file

Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""
import rasterio
import re
import time,datetime
import argparse
from rasterio import mask,features,warp
import os
import os.path
import fiona
import numpy as np
import osgeo
from osgeo import gdal,osr
from shapely.geometry import shape,mapping
from shapely.geometry.polygon import LinearRing,Polygon
from mfuncshape import *
from PIL import Image
from m_util import sdmkdir, convertMbandstoRGB,sdsaveim
import pandas as pd
from shutil import copyfile


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opt = parser.parse_args()
padding  = 300
opt.root = '/gpfs/projects/LynchGroup/'
opt.resdir = '/gpfs/projects/LynchGroup/Penguin_workstation/data/Penguins' + '/TEST_PTS_MASK_PADDING_' + str(padding) + '/' 
opt.A = opt.resdir + 'A/'
opt.B = opt.resdir + 'B/'
opt.tif_fold = opt.root + 'Orthoed/'
sdmkdir(opt.resdir)
sdmkdir(opt.A)
sdmkdir(opt.B)
opt.shape_dir = opt.root+ '/Annotated_shapefiles_PTS/'
files= []
for root,_,fnames in sorted(os.walk(opt.shape_dir)):
    for fname in fnames:
        if fname.endswith('tif'):
            files.append(fname)
for file1 in files:
    indx = file1.find('__')
    file2 = file1[indx+2:]
    print(file2)
    match = re.search(r'\d{2}\D{3}\d{8}', file2).group(0)

    date = '20'+match[0:2] +  "%02d"%(time.strptime(match[2:5],'%b').tm_mon)+match[5:]
    date = datetime.datetime.strptime(date, '%Y%m%d%H%M%S')
    
    
    if date.year ==2017 or date.year==2018:
        TIF1 = opt.tif_fold + str(date.year)+'/' + "%02d"%date.month +'/'+file2
    else:
        TIF1 = opt.tif_fold + str(date.year) +'/' +file2

    TIF2 = opt.shape_dir + file1
    Im1 = rasterio.open(TIF1)
    Im2 = rasterio.open(TIF2)
    print(Im1.meta)
    print(Im2.meta)

    out_meta = Im1.meta.copy()
    out_meta.update({"count":out_meta["count"]+1
                    })
    X = Im1.read()
    print(X.shape)
    GT = Im2.read()
    GT = GT[0:1,:,:]

    x1,y1 = np.where(GT[0,:,:]!=255)
    maxx = np.max(x1) + padding
    minx = np.min(x1) - padding
    maxy = np.max(y1) + padding
    miny = np.min(y1) - padding
    im = X[:,minx:maxx,miny:maxy]
    im = convertMbandstoRGB(im,file2)
    mask = GT[:,minx:maxx,miny:maxy]
    mask[mask!=255] = 1
    mask[mask==255] = 0
    mask[mask==1] = 255
    print(im.shape)
    print(mask.shape)
    im = np.transpose(im,(1,2,0))
    mask = np.transpose(mask,(1,2,0))
    mask = np.squeeze(mask)
    sdsaveim(im,opt.A+file2.replace('.tif','.png'))
    sdsaveim(mask,opt.B + file2.replace('.tif','.png'))

    #X_and_GT = np.concatenate((X,GT),axis=0)
    
    #with rasterio.open( opt.resdir+file2,"w",**out_meta) as dest:
    #    dest.write(X_and_GT)
