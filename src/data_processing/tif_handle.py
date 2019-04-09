"""
Class for handling reading/writting TIF files

Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""
import rasterio
import numpy as np
import os

class TIF_H:
    def __init__(self,name):
        self.name = name
    def get_png(self):
        self.src = rasterio.open(self.name)
        tif = self.src.read()
        tif = tif.astype(np.float32)
        self.png = self.convertMbandstoRGB(tif)
        self.png = np.transpose(self.png,(1,2,0))
        for i in range(3):
            maxi = np.amax(self.png[:,:,i])
            mini = np.amin(self.png[:,:,i])
            self.png[:,:,i] =  (self.png[:,:,i] - mini)/(maxi-mini)*255
            print(maxi,mini)
        self.profile = self.src.profile
    def convertMbandstoRGB(self,tif):
#        try:
        if tif.shape[0] ==1:
            return tif
        imname = os.path.basename(self.name)
        if "QB" in imname:
            return tif[(3,2,1),:,:]
        if "WV" in imname:
            if tif.shape[0] ==8:
                return tif[(5,3,2),:,:]
            if tif.shape[0] ==4:
                return tif[(3,2,1),:,:]
        if "IK" in imname:
            return tif[(3,2,1),:,:]
 #       except:
  #          print('error while reading file')
