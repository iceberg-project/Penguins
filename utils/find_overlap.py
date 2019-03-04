__all__ = ['find_overlap']

import argparse

import geopandas as gpd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(prog='Finds which points fall within shapefiles')
    parser.add_argument('--footprint_shp', type=str, help='path to footprint shapefile for rasters')
    parser.add_argument('--points_shp', type=str, help='path to shapefile with points')
    parser.add_argument('--id_columns', type=str,
                        help='ID columns to store matches, separated by - (e.g. footprintID-pointID')
    parser.add_argument('--min_dist', type=float, help='minimum distance between point and border')
    parser.add_argument('--out_file', type=str, help='output file name')
    parser.add_argument('--n_workers', type=int, help='number of workers for multiprocessing pool')
    return parser.parse_args()


def find_overlap(footprint_shp, points_shp, min_dist, id_columns, output_name, n_workers):
    """
    Checks if points falls inside a raster on footprint shapefile. Distance to border can be used to ensure rasters
    containing pointss can be cropped to given size.

    :param footprint_shp:
    :param points_shp:
    :param id_columns:
    :param min_dist:
    :param output_name:
    :param n_workers:
    :return:
    """
    # get id_columns
    id_columns = id_columns.split('-')

    # read files
    footprint = gpd.read_file(footprint_shp)
    points = gpd.read_file(points_shp)

    # check if point and shapefile are using the same CRS
    if footprint.crs != points.crs:
        points = points.to_crs(footprint.crs)

    # get footprint entries that are most reliable for guano detection (DEC 15 - JAN 15)
    footprint = footprint[np.array(footprint.dayofseaso > 197) * np.array(footprint.dayofseaso < 229)]

    # get only multispectral scenes
    footprint = footprint[footprint['location'].str.contains('M1BS')]

    # buffer points
    points_bfr = points.buffer(min_dist)

    # matching function
    def match(scene, points, scene_id, point_id):
        out = {'scene': [], 'site': [], 'geometry': [], 'year': [], 'path': []}
        for idx, point in points.iterrows():
            if points_bfr.iloc[idx].within(scene.geometry):
                out['scene'].append(scene[scene_id])
                out['site'].append(point[point_id])
                out['geometry'].append(point.geometry)
                out['year'].append(scene.year)
                out['path'].append(scene.location)
        return out

    # iter through items using multiprocessing pool
    scenes = [scene for _, scene in footprint.iterrows()]
    pool = Pool(n_workers)
    out = pool.map(partial(match, points=points, scene_id=id_columns[0], point_id=id_columns[1]), scenes)
    pool.close()
    pool.join()

    # aggregate output
    out_fixed = {'scene': [], 'site': [], 'geometry': [], 'year': [], 'path': []}
    for ele in out:
        for key in ele:
            out_fixed[key].extend(ele[key])

    # geopandas dataframe to store output in shapefile
    matches = gpd.GeoDataFrame(data=out_fixed, crs=points.crs)

    # save output to csv
    matches.to_file(output_name)


def main():
    args = parse_args()
    find_overlap(footprint_shp=args.footprint_shp,
                 points_shp=args.points_shp,
                 min_dist=args.min_dist,
                 id_columns=args.id_columns,
                 output_name=args.out_file,
                 n_workers=args.n_workers)


if __name__ == "__main__":
    main()
