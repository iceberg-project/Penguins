import fiona
import numpy as np
import os
from rasterio import mask,features,warp
from mfuncshape import ReadProjection,TransformShape
from shapely.geometry.polygon import Polygon
import rasterio
from osgeo import gdal, osr
from PIL import Image
import math
import datetime
from datetime import datetime
class dataset():
    def __init__(self,shapefile):
        self.name =shapefile
        

def shannon_entropy(img):
    img = Image.fromarray(img)
    histogram = img.histogram()
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]

    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

def normalizeRGB(savetif):
    savetif = np.transpose(savetif,(1,2,0))
    if savetif.dtype == np.uint16:
        savetif = savetif.astype(np.float)
        for i in range(0,savetif.shape[2]):
            savetif[:,:,i] =  (savetif[:,:,i] - np.min(savetif[:,:,i])) / (np.max(savetif[:,:,i])-np.min(savetif[:,:,i])) * 255
        savetif = savetif.astype(np.uint8) 
    return savetif
def convertMbandstoRGB(tif,imname):
    if tif.shape[0] ==1:
        return normalizeRGB(tif)
    if "QB" in imname:
        return normalizeRGB(tif[(3,2,1),:,:])
    if "WV" in imname:
        if tif.shape[0] ==8:
            return normalizeRGB(tif[(5,3,2),:,:])
        if tif.shape[0] ==4:
            return normalizeRGB(tif[(3,2,1),:,:])
    if "IK" in imname:
        return normalizeRGB(tif[(3,2,1),:,:])

def to_rgb3b(im):
    # as 3a, but we add an extra copy to contiguous 'C' order
    # data
    # ... where is to_rgb3a?
    return np.dstack([im.astype(np.uint8)] * 3).copy(order='C')
def sdsaveim(savetif,name):
    print(name)
    if len(savetif.shape)== 2:
        Image.fromarray(savetif.astype(np.uint8),mode='L').save(name)
    elif savetif.shape[2] == 3:
        Image.fromarray(savetif.astype(np.uint8)).save(name)

def sdmkdir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

def contains_all(P,set_of_P):
    for k in set_of_P:
        if not P.contains(k):
            return False
    return True
def main():
    padding = 400
    gt_dir  = '/gpfs/projects/LynchGroup/Penguin_workstation/data/Penguins/Reprojected_shapefiles/' 
    allf = []
    for file in os.listdir(gt_dir):
        if file.endswith('.shp') and not file.startswith('.'):
            allf.append(file[:-4])
    for filename in allf:
    #filename ='RP_crozier_guanoarea'
        s1 = gt_dir + '/' + filename + '.shp' 
        gt_prj = ReadProjection(gt_dir + '/' + filename+'.prj') 
        s2 ='/gpfs/projects/LynchGroup/Penguin_workstation/data//footprint_Feb24th/Orthoed.shp'

        save_path= '/gpfs/projects/LynchGroup/Penguin_workstation/data/Penguins/Cropped_ALL_footprint_shannon5_padding' + str(padding)
        gt = fiona.open(s1)
        raster = fiona.open(s2)
        x = []
        for ft in gt:
            x.append(Polygon(ft['geometry']['coordinates'][0]))

        im_list = [];
        for im in raster:
            im_polygon = Polygon(im['geometry']['coordinates'][0])
            datestr = im['properties']['date']
            date = datetime.strptime(datestr,'%Y-%m-%d %H:%M:%S')
    #        if date.month>2 and date.month<10:
    #            continue
    #        if date.month == 12 and date.day<15:
    #            continue
    #        if date.month == 1 and date.day>15:
    #            continue

            if 'M1BS' in im['properties']['location']:
                if contains_all(im_polygon,x):
                    im_list.append(im['properties']['location'])
        print(len(im_list))
        count=0
        for TIFim in im_list:
            try:
                if 'M1BS' in TIFim:
                    im = rasterio.open(TIFim)
                    imname = os.path.basename(TIFim)
                    im_prj = ReadProjection(TIFim.replace('.tif','.prj'))
                    coordTrans = osr.CoordinateTransformation(gt_prj,im_prj)
                    transformed_gt,bb = TransformShape(gt,coordTrans,padding = padding)
                    crop_image, ct = rasterio.mask.mask(im,[po['geometry'] for po in bb],crop=True)


                    
                    out_meta = im.meta.copy()
                    out_meta.update({"driver":"GTiff",
                                    "height": crop_image.shape[1],
                                    "width": crop_image.shape[2],
                                    "transform": ct
                                    })
                    with rasterio.open('temp.tif',"w",**out_meta) as dest:
                        dest.write(crop_image)
                    dest = rasterio.open( 'temp.tif')
                    
                    masked_image, mt = rasterio.mask.mask(dest,[feature["geometry"] for feature in transformed_gt])
                
                    mask = masked_image.mean(axis=0)
                    mask[mask>0]=255
                    mask[mask<255]= 0 
                    crop_image = convertMbandstoRGB(crop_image,imname)
                    entropy=shannon_entropy(crop_image) 
                    if entropy<5:
                        print('FAILED VALUE TEST')
                        continue
                    count = count+1      
                    sdmkdir(os.path.join(save_path,filename,'A')) 
                    sdmkdir(os.path.join(save_path,filename,'B')) 
                    sdsaveim(crop_image,os.path.join(save_path,filename,'A',imname+'.png'))
                    sdsaveim(mask,os.path.join(save_path,filename,'B',imname+'.png'))
            except:
                print('FAILED')
                pass

            
         #   Image.fromarray(crop_image.astype(np.uint8)
        print(count)

if __name__ == '__main__':
    main()

    
