__all__ = ['find_overlap']

import argparse

import geopandas as gpd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog='Finds which points fall within shapefiles')
    parser.add_argument('--footprint_shp', type=str, help='path to footprint shapefile for rasters')
    parser.add_argument('--points_shp', type=str, help='path to shapefile with points')
    parser.add_argument('--id_columns', type=str,
                        help='ID columns to store matches, separated by - (e.g. footprintID-pointID')
    parser.add_argument('--min_dist', type=float, help='minimum distance between point and border')
    parser.add_argument('--out_file', type=str, help='output file name')
    return parser.parse_args()


def find_overlap(footprint_shp, points_shp, min_dist, id_columns, output_name):
    """
    Checks if points falls inside a raster on footprint shapefile. Distance to border can be used to ensure rasters
    containing pointss can be cropped to given size.

    :param footprint_shp:
    :param points_shp:
    :param id_columns:
    :param min_dist:
    :param output_name:
    :return:
    """
    # get id_columns
    id_columns = id_columns.split('-')

    # read files
    footprint = gpd.read_file(footprint_shp)
    points = gpd.read_file(points_shp)

    # check if point and shapefile are using the same CRS
    assert footprint.crs == points.crs, 'Footprint and points in different coordinate systems.'

    # geopandas dataframe to store output in shapefile
    matches = gpd.GeoDataFrame(crs=points.crs)

    # get footprint entries that are most reliable for guano detection (DEC 15 - JAN 15)
    footprint = footprint[np.array(footprint.dayofseaso > 197) * np.array(footprint.dayofseaso < 229)]

    # buffer points
    points_bfr = points.buffer(min_dist)

    # iter through items
    for idx, point in points.iterrows():
        for idx2, scene in footprint.iterrows():
            if points_bfr.iloc[idx].within(scene.geometry):
                matches = matches.append({'scene': scene[id_columns[0]],
                                          'site': point[id_columns[1]],
                                          'geometry': point.geometry,
                                          'year': scene.year,
                                          'path': scene.location},
                                         ignore_index=True)

    # save output to csv
    matches.to_file(output_name)


def main():
    args = parse_args()
    find_overlap(footprint_shp=args.footprint_shp,
                 points_shp=args.points_shp,
                 min_dist=args.min_dist,
                 id_columns=args.id_columns,
                 output_name=args.out_file)


if __name__ == "__main__":
    main()
