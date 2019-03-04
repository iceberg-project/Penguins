import rasterio
from rasterio.warp import transform_bounds
from rasterio import windows
import cv2
import argparse
import os
import numpy as np
import re
import geopandas as gpd

__all__ = ['get_crop']


def parse_args():
    parser = argparse.ArgumentParser(prog='Crops training patches from rasters')
    parser.add_argument('--rasters_dir', type=str, help="path to folder with input rasters")
    parser.add_argument('--selection_shp', type=str, help="path to shapefile with selected rasters")
    parser.add_argument('--out_dir', type=str, help="folder to save cropped rasters")
    return parser.parse_args()


def get_crop(raster, location, pnt_crs, crop_size, out):
    """
    Helper function to crop raster at location using a predefined crop_size

    :param raster:
    :param location:
    :param crop_size:
    :return:
    """

    # open raster and crop
    with rasterio.open(raster) as src:

        # get number of bands for raster
        shp = src.shape[0]
        src_crs = src.crs

        # check if point and raster share the same CRS
        if pnt_crs != src_crs:
            bbox = location['geometry'].buffer(0.005).bounds
            bbox = transform_bounds(pnt_crs, src_crs, *bbox)

        else:
            bbox = location['geometry'].buffer(0.005).bounds

        # extract window
        window = windows.from_bounds(*bbox, src.transform)
        window.height = crop_size
        window.width = crop_size

        try:
            # get crop
            cropped = src.read(window=window)

            # get rgb channels
            sensor = re.search(r'[A-Z]{2}\d{2}', raster.split('/')[-1]).group(0)

            if 'WV' in sensor:
                if shp == 8:
                    cropped = cropped[(5, 3, 2), :, :]
                else:
                    cropped = cropped[(3, 2, 1), :, :]
            else:
                cropped = cropped[(3, 2, 1), :, :]

            # write to png -- careful with RGB channels
            if cropped.shape[1:] == (crop_size, crop_size) and np.max(cropped) > 0:
                for i in range(cropped.shape[0]):
                    cropped[i, :, :] = cropped[i, :, :] / np.max(cropped[i, :, :]) * 255
                cv2.imwrite(out, cropped.transpose([2, 1, 0]))

        except Exception as exc:
            print(f'location out of bounds for {raster}')
            print(exc)


def main():
    # unroll arguments
    args = parse_args()
    rasters_dir = args.rasters_dir
    out_dir = args.out_dir
    points = gpd.read_file(args.selection_shp)

    # find rasters
    rasters = [file for file in os.listdir(rasters_dir) if file.split('.')[-1] == 'tif']

    # loop over rasters
    for raster in rasters:
        locations = points.loc[points['scene'] == raster]
        for _, row in locations.iterrows():
            get_crop(f"{rasters_dir}/{raster}", row, points.crs, crop_size=702,
                     out=f"{out_dir}/{raster}_{row['site']}.png")


if __name__ == "__main__":
    main()
