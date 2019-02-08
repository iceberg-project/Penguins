import rasterio
import argparse
from rasterio import mask,features,warp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
from matplotlib import cm
from m_util import sdmkdir, convertMbandstoRGB,sdsaveim
import pandas as pd
from shutil import copyfile


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opt = parser.parse_args()
opt.root = '/gpfs/projects/LynchGroup/'
opt.resdir = opt.root + '/TEST_PTS_MASK/'
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
    print file2
    TIF1 = opt.tif_fold+file2
    TIF2 = opt.shape_dir + file1
    Im1 = rasterio.open(TIF1)
    Im2 = rasterio.open(TIF2)
    print(Im1.meta)
    print(Im2.meta)

    out_meta = Im1.meta.copy()
    out_meta.update({"count":out_meta["count"]+1
                    })
    X = Im1.read()
    GT = Im2.read()
    print X.shape
    print GT.shape
    GT = GT[0:1,:,:]
    print np.unique(GT)

    x1,y1 = np.where(GT[0,:,:]!=255)
    padding  = 1000
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
    im = np.transpose(im,(1,2,0))
    mask = np.transpose(mask,(1,2,0))
    sdsaveim(im,opt.A+file2.replace('.tif','.png'))
    sdsaveim(mask,opt.B + file2.replace('.tif','.png'))

    #X_and_GT = np.concatenate((X,GT),axis=0)
    
    #with rasterio.open( opt.resdir+file2,"w",**out_meta) as dest:
    #    dest.write(X_and_GT)
