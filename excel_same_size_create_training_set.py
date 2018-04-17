import scipy.misc as misc
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
opt.padding = 500

files = ['/gpfs/projects/LynchGroup/PAUL_IDs_Test.xlsx','/gpfs/projects/LynchGroup/CROZ_IDs_Test.xlsx',
            '/gpfs/projects/LynchGroup/CatalogIDs_training_shapefiles.xlsx']
folds = ['Test/PAUL/','Test/CROZ/','Train/Train_all/']
for id in range(0,3):
    file = files[id]
    opt.fold =  folds[id]
    opt.root = '/gpfs/projects/LynchGroup/'

    opt.raw_fold = opt.root + opt.fold+ '/raw/'
    opt.tif_fold = opt.root + 'Orthoed/'
    opt.training_fold = opt.root + 'MB_Same_Size/' + opt.fold+ '/CROPPED/p'+str(opt.padding)+'/'
    opt.A = opt.training_fold + 'A/'
    opt.B = opt.training_fold + 'B/'
    opt.visdir = opt.training_fold + '/vis/'
    opt.ctifdir = opt.training_fold + '/tif/'
    sdmkdir(opt.training_fold)
    sdmkdir(opt.A)
    sdmkdir(opt.B)
    sdmkdir(opt.ctifdir)

    #shape_dir= '/gpfs/projects/LynchGroup/Colony\ shapefiles\ from\ imagery/'
    shape_dir = opt.root+ '/Annotated_shapefiles/'

    anno = pd.read_excel(file,sheet_name=0)
    tif = anno['Filename']
    shape =  anno['Shapefile of guano']
    for i in range(0,len(tif)):
        name= tif[i].encode('ascii','ignore')
        if '-M' in name:
            gta= shape[i].encode('ascii','ignore')
            name = name.replace('.tif','') 
            gt_prj = ReadProjection(shape_dir+gta+'.prj')
            gt = fiona.open(shape_dir+gta+'.shp')
            
            name= name.replace('-M','-*****')
            imname_P= name.replace('-*****','-P')
            imname_M= name.replace('-*****','-M')
            TIFim_P = opt.tif_fold+imname_P+'.tif'
            TIFim_M = opt.tif_fold+imname_M+'.tif'
            if os.path.isfile(TIFim_P):
                PBANDIMG = rasterio.open(TIFim_P)
                im_prj = osgeo.osr.SpatialReference()
                im_prj.ImportFromWkt(PBANDIMG.crs.wkt)
                coordTrans = osr.CoordinateTransformation(gt_prj,im_prj)
                transformed_gt,bb = TransformShape(gt,coordTrans,padding = opt.padding)
                crop_image, ct = rasterio.mask.mask(PBANDIMG,[po['geometry'] for po in bb],crop=True)
                out_meta = PBANDIMG.meta.copy()
                out_meta.update({"driver":"GTiff",
                                "height": crop_image.shape[1],
                                "width": crop_image.shape[2],
                                "transform": ct
                                })
                with rasterio.open(opt.ctifdir+imname_P+'.tif',"w",**out_meta) as dest:
                    dest.write(crop_image)
                
                dest = rasterio.open( opt.ctifdir+imname_P+'.tif')
                masked_image, mt = rasterio.mask.mask(dest,[feature["geometry"] for feature in transformed_gt])
                mask = masked_image.mean(axis=0)
                mask[mask>0]=255
                mask[mask<255]= 0 
                Image.fromarray(mask.astype(np.uint8)).save(opt.B+'/'+imname_P+'.png',cmap=cm.gray)
                tifimg = crop_image
                tifimg = convertMbandstoRGB(tifimg,imname_P)
                savetif = tifimg
                savetif = np.transpose(savetif,(1,2,0))
                sdsaveim(savetif,opt.A+'/'+imname_P+'.png')
                Pim_size = tifimg.shape
                
                
                
                MBANDIMG = rasterio.open(TIFim_M)
                crop_image_pband,ct = rasterio.mask.mask(MBANDIMG,[po['geometry'] for po in bb],crop=True)
                out_meta = MBANDIMG.meta.copy()
                out_meta.update({"driver":"GTiff",
                                "height": crop_image_pband.shape[1],
                                "width": crop_image_pband.shape[2],
                                "transform": ct
                                })
                with rasterio.open(opt.ctifdir+imname_M+'.tif',"w",**out_meta) as dest:
                    dest.write(crop_image_pband)

                dest = rasterio.open( opt.ctifdir+imname_P+'.tif')
                #PURPOSEDLY READ THIS P-BAND IMAGE SINCE THEY SHOULD BE AT THE SAME SIZE AS M-BAND NOW. MIGHT THIS LEAD TO A DISASTER LATER?
                masked_image, mt = rasterio.mask.mask(dest,[feature["geometry"] for feature in transformed_gt])
                mask = masked_image.mean(axis=0)
                mask[mask>0]=255
                mask[mask<255]= 0 
                Image.fromarray(mask.astype(np.uint8)).save(opt.B+'/'+imname_M+'.png',cmap=cm.gray)
                tifimg = crop_image_pband
                tifimg = convertMbandstoRGB(tifimg,imname_M)
                savetif = tifimg
                savetif = np.transpose(savetif,(1,2,0))
                savetif = misc.imresize(savetif,Pim_size[1:])
                sdsaveim(savetif,opt.A+'/'+imname_M+'.png')
                
                #copyfile(opt.tif_fold+imname+'.prj',opt.A+'/'+imname+'.prj')
                #copyfile(opt.tif_fold+imname+'.tif',ndir+imname+'.tif')
                #copyfile(opt.tif_fold+imname+'.prj',ndir+imname+'.prj')

