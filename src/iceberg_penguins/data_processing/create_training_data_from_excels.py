"""
Extract the annotations from the excelsheet and polygon shapefiles

Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""
import re,time,datetime
import rasterio
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
opt.padding = 300

opt.root = '/gpfs/projects/LynchGroup/Penguin_workstation/data/Penguins/'
shape_dir = '/gpfs/projects/LynchGroup/Annotated_shapefiles/'
opt.tif_fold = '/gpfs/projects/LynchGroup/Orthoed/'
files = ['PAUL_IDs_Test.xlsx','CROZ_IDs_Test.xlsx',
            'CatalogIDs_training_shapefiles.xlsx']
folds = ['Test/PAUL/','Test/CROZ/','Train_all/']
for id in range(0,3):
    file = opt.root+files[id]
    opt.fold =  folds[id]

    opt.training_fold = opt.root + opt.fold+ '/padding_'+str(opt.padding)+'/'
    opt.A = opt.training_fold + 'A/'
    opt.B = opt.training_fold + 'B/'

    opt.ctifdir = opt.root + opt.fold+ '/padding_' +str(opt.padding)+ '/tif/'

    sdmkdir(opt.training_fold)
    sdmkdir(opt.A)
    sdmkdir(opt.B)
    sdmkdir(opt.ctifdir)

    #shape_dir= '/gpfs/projects/LynchGroup/Colony\ shapefiles\ from\ imagery/'

    anno = pd.read_excel(file,sheet_name=0)
    tif = anno['Filename']
    shape =  anno['Shapefile of guano']
    for i in range(0,len(tif)):
        name= tif[i].encode('ascii','ignore')
        name = name.decode()
        if "-M" in name:
            gta= shape[i].encode('ascii','ignore')
            gta = gta.decode()
            name = name.replace('.tif','') 
            gt_prj = ReadProjection(shape_dir+gta+'.prj')
            gt = fiona.open(shape_dir+gta+'.shp')
            name= name.replace('-M','-*****')
            name= name.replace('-P','-*****')
            for BAND in ['-M']:
                imname= name.replace('-*****',BAND)
                print(imname)
                match = re.search(r'\d{2}\D{3}\d{8}', imname).group(0)

                date = '20'+match[0:2] +  "%02d"%(time.strptime(match[2:5],'%b').tm_mon)+match[5:]
                date = datetime.datetime.strptime(date, '%Y%m%d%H%M%S')
                
                
                if date.year ==2017 or date.year==2018:
                    TIF1 = opt.tif_fold + str(date.year)+'/' + "%02d"%date.month +'/'+imname
                else:
                    TIF1 = opt.tif_fold + str(date.year) +'/' +imname
                
                TIFim = TIF1+'.tif'
                
                MBandImg = rasterio.open(TIFim)
                im_prj = osgeo.osr.SpatialReference()
                im_prj.ImportFromWkt(MBandImg.crs.wkt)
                coordTrans = osr.CoordinateTransformation(gt_prj,im_prj)
                transformed_gt,bb = TransformShape(gt,coordTrans,padding = opt.padding)

                crop_image, ct = rasterio.mask.mask(MBandImg,[po['geometry'] for po in bb],crop=True)

                out_meta = MBandImg.meta.copy()
                out_meta.update({"driver":"GTiff",
                                "height": crop_image.shape[1],
                                "width": crop_image.shape[2],
                                "transform": ct
                                })
                with rasterio.open(opt.ctifdir+imname+'.tif',"w",**out_meta) as dest:
                    dest.write(crop_image)
                dest = rasterio.open( opt.ctifdir+imname+'.tif')
                
                masked_image, mt = rasterio.mask.mask(dest,[feature["geometry"] for feature in transformed_gt])
                mask = masked_image.mean(axis=0)
                mask[mask>0]=255
                mask[mask<255]= 0 
                print(opt.B+'/'+imname+'.png')
                sdsaveim(mask,opt.B+'/'+imname+'.png')
                tifimg = crop_image
                tifimg = convertMbandstoRGB(tifimg,imname)
                savetif = tifimg
                savetif = np.transpose(savetif,(1,2,0))
                sdsaveim(savetif,opt.A+'/'+imname+'.png')

