import rasterio
import shapely
from rasterio import windows
from skimage.measure import shannon_entropy
from scipy import ndimage
import cv2
import argparse
import os
import random
import numpy as np
import re
import geopandas as gpd
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial

__all__ = ['get_crops']


def parse_args():
    parser = argparse.ArgumentParser(prog='Crops training patches from rasters')
    parser.add_argument('--rasters_dir', type=str, help="path to folder with input rasters")
    parser.add_argument('--selection_shp', type=str, help="path to shapefile with selected rasters")
    parser.add_argument('--polygon_shp', type=str, help='shapefile with polygons to be rasterized as groundtruth masks')
    parser.add_argument('--pad', type=int, default=250, help="pad width for crops")
    parser.add_argument('--out_dir', type=str, help="folder to save cropped rasters")
    parser.add_argument('--patch_size', default=256, type=int, help="sliding window patch size")
    parser.add_argument('--stride', default=128, type=int, help="stride for sliding window")
    parser.add_argument('--shannon_thresh', default=0, type=int, help="shannon threshold to take crops")
    parser.add_argument('--cores', default=2, type=int, help="number of cores for multiprocessing pool")
    return parser.parse_args()


def write_window(window, src, shannon_thresh, pad):
    """

    :param window:
    :param src:
    :param out_dir:
    :param idx:
    :return:
    """

    # get number of bands for raster
    n_bands = src.count
    raster = src.name.split('/')[-1]

    try:
        # add buffer to window matching shape padding
        window.row_off -= pad // 2
        window.col_off -= pad // 2
        window.height += pad
        window.width += pad

        # get
        cropped = src.read(window=window)

        # get rgb channels
        sensor = re.search(r'[A-Z]{2}\d{2}', raster.split('/')[-1]).group(0)

        if sensor[:2] == 'WV':
            if n_bands == 8:
                cropped = cropped[(2, 3, 5), :, :]
            else:
                cropped = cropped[(1, 2, 3), :, :]
        else:
            cropped = cropped[(1, 2, 3), :, :]

        # normalize and change dtype to uint8
        for i in range(cropped.shape[0]):
            min_val = np.amin(cropped[i, :, :])
            max_val = np.amax(cropped[i, :, :])
            cropped[i, :, :] = (cropped[i, :, :] - min_val) / (max_val - min_val) * 255
        cropped = cropped.transpose([2, 1, 0])
        cropped = cropped.astype(np.uint8)

        # get shannon index and save
        if shannon_entropy(cropped) > shannon_thresh:
            return True, cropped
        else:
            return False, None

    except Exception as exc:
        print(f'location out of bounds for {raster}')
        print(src.shape)
        print(window)
        print(exc)
        return False, None


def rasterize_pol(pol, shape, pad):
    """
    Helper function to raseterize polygon
    :param pol:
    :param out_dir:
    :return:
    """
    # remove padding from windows
    height, width = [ele - pad for ele in shape]

    def get_shortest_path(pnt1, pnt2):
        # initiate empty path
        path = [pnt1]

        # difference between pnt1 and pnt2
        diff_x = pnt1[0] - pnt2[0]
        diff_y = pnt1[1] - pnt2[1]

        # direction to go on axes x and y
        dir_x = 1 if diff_x <= 0 else -1
        dir_y = 1 if diff_y <= 0 else -1

        # get moves
        moves = ['x'] * abs(diff_x) + ['y'] * abs(diff_y)
        random.shuffle(moves)

        for move in moves:
            curr = path[-1]
            if move == 'x':
                path.append([curr[0] + dir_x, curr[1]])
            else:
                path.append([curr[0], curr[1] + dir_y])

        return path

    # set output array
    out = np.zeros([height, width], dtype=np.uint8)

    # get minimum and maximum values for scaling
    minx, miny, maxx, maxy = [np.array(ele) for ele in pol['geometry'].bounds]

    # check for multi-polygon
    if type(pol['geometry']) == shapely.geometry.multipolygon.MultiPolygon:

        for geo in pol['geometry']:
            # get coordinates
            x, y = geo.exterior.xy

            # rescale to fit shape indeces
            x = [(height - 1) - int(round(ele)) for ele in (x - minx) / (maxx - minx) * (height - 1)]
            y = [(width - 1) - int(round(ele)) for ele in (y - miny) / (maxy - miny) * (width - 1)]

            # add boundary to output raster
            for idx in range(len(x) - 1):
                path = get_shortest_path(pnt1=[x[idx], y[idx]], pnt2=[x[idx + 1], y[idx + 1]])
                out[[ele[0] for ele in path], [ele[1] for ele in path]] = 255

            # fill interior

    else:
        # get coordinates
        x, y = pol['geometry'].exterior.xy

        # rescale coordinates to fit shape indeces (indexation is different on np and rasterio)
        x = [(height - 1) - int(round(ele)) for ele in (x - minx) / (maxx - minx) * (height - 1)]
        y = [(width - 1) - int(round(ele)) for ele in (y - miny) / (maxy - miny) * (width - 1)]

        # add boundary to output raster
        for idx in range(len(x) - 1):
            path = get_shortest_path(pnt1=[x[idx], y[idx]], pnt2=[x[idx + 1], y[idx + 1]])
            out[[ele[0] for ele in path], [ele[1] for ele in path]] = 255

        # fill interior

    # write to file
    out = np.pad(out, pad_width=pad // 2, mode='constant')
    out = np.flipud(out)
    out[ndimage.binary_fill_holes(out)] = 255
    return out


def extract_patches(input_crop, input_pol, patch_size, stride, out_dir, fname):
    """

    :param input_crop:
    :param input_pol:
    :param patch_size:
    :param stride:
    :param out_dir:
    :param fname:
    :return:
    """

    for x_idx in range(0, input_pol.shape[0] - 1 - patch_size, stride):
        for y_idx in range(0, input_pol.shape[1] - 1 - patch_size, stride):
            pol_patch = input_pol[x_idx:x_idx + patch_size, y_idx: y_idx + patch_size]
            crop_patch = input_crop[x_idx:x_idx + patch_size, y_idx: y_idx + patch_size, :]
            if np.sum(pol_patch) > 0:
                cv2.imwrite(f"{out_dir}/guano-x/{x_idx}_{y_idx}_{fname}", crop_patch)
                cv2.imwrite(f"{out_dir}/guano-y/{x_idx}_{y_idx}_{fname}", pol_patch)
            else:
                cv2.imwrite(f"{out_dir}/background/{x_idx}_{y_idx}_{fname}", crop_patch)


def get_crops(raster, selection, polygons, stride, shannon_thresh, patch_size, pol_crs, pad, out_dir, train_val=0.9):
    """
    Helper function to crop raster at location using a predefined crop_size

    :param raster:
    :param location:
    :param crop_size:
    :return:
    """

    # create output folders
    if random.random() > train_val:
        split = 'validation'
    else:
        split = 'training'

    for dir in ['guano-x', 'guano-y', 'background']:
        os.makedirs(f"{out_dir}/{split}/{dir}", exist_ok=True)

    # open raster and crop
    with rasterio.open(raster) as src:
        src_crs = src.crs
        src.profile.update(dtype=rasterio.uint8)

        # check if point and raster share the same CRS and create bboxes
        if pol_crs != src_crs:
            selection = selection.to_crs(src_crs)

        bboxes = [geo.bounds for geo in selection.geometry]
        pol_idcs = [idx for idx in selection.pol_idx]

        # extract window
        for idx, bbox in enumerate(bboxes):
            window = windows.from_bounds(*bbox, src.transform)
            pol = polygons.iloc[pol_idcs[idx]]

            fname = f"{raster.split('/')[-1].split('.')[0]}_{pol['Code']}_{idx}.png"
            succeeded, out_crop = write_window(window, src, shannon_thresh, pad)
            if succeeded:
                out_pol = rasterize_pol(pol, (int(window.width), int(window.height)), pad)

                extract_patches(input_crop=out_crop, input_pol=out_pol, patch_size=patch_size,
                                out_dir=f"{out_dir}/{split}",
                                stride=stride, fname=fname)


def main():
    # unroll arguments
    args = parse_args()
    rasters_dir = args.rasters_dir
    out_dir = args.out_dir
    selection = gpd.read_file(args.selection_shp)
    pols = gpd.read_file(args.polygon_shp)
    shannon_thresh = args.shannon_thresh
    pad = args.pad
    patch_size = args.patch_size
    cores = args.cores
    stride = args.stride

    # find rasters
    rasters = []
    locations = []
    for path, _, files in os.walk(rasters_dir):
        rasters += [f'{path}/{file}' for file in files if file.endswith('.tif')]
        locations += [selection.loc[selection['scene'] == file] for file in files if file.endswith('.tif')]

    # start threading pool
    pool = Pool(nodes=cores)

    import time
    tic = time.time()
    get_crops_pool = partial(get_crops, shannon_thresh=shannon_thresh,
                             patch_size=patch_size,
                             polygons=pols,
                             stride=stride,
                             pol_crs=pols.crs,
                             pad=pad,
                             out_dir=out_dir)

    pool.map(get_crops_pool, rasters, locations)
    toc = time.time()
    print('time elapsed :', (toc - tic) / 60, 'minutes')


if __name__ == "__main__":
    main()
