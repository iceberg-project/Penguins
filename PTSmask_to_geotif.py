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


file1 = 'Guano_WV02_20151204195602_103001004F9A8500_15DEC04195602-M1BS-500637515080_01_P006_u08rf3031.tif'
file2 = file1.replace('Guano_','')
print(file2)
file1 = 'WV02_20151204195602_103001004F9A8500_15DEC04195602-M1BS-500637515080_01_P006_u08rf3031.tif'
file2 = file1
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opt = parser.parse_args()
opt.root = '/gpfs/projects/LynchGroup/'
opt.resdir = opt.root + '/TEST_PTS_MASK/'
opt.tif_fold = opt.root + 'Orthoed/'
sdmkdir(opt.resdir)
opt.shape_dir = opt.root+ '/Annotated_shapefiles_PTS/'
files= []
for root,_,fnames in sorted(os.walk(opt.shape_dir)):
    for fname in fnames:
        if fname.endswith('tif'):
            files.append(fname)
for file1 in files:
    file2 = file1
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
    X_and_GT = np.concatenate((X,GT),axis=0)

    with rasterio.open( opt.resdir+file2,"w",**out_meta) as dest:
        dest.write(X_and_GT)
