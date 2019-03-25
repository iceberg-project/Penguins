__all__ = ['find_overlap']

import argparse

import geopandas as gpd
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from shapely.geometry.geo import box


def parse_args():
    parser = argparse.ArgumentParser(prog='Finds which points fall within shapefiles')
    parser.add_argument('--footprint_shp', type=str, help='path to footprint shapefile for rasters')
    parser.add_argument('--pol_shp', type=str, help='path to shapefile with points')
    parser.add_argument('--id_columns', type=str,
                        help='ID columns to store matches, separated by - (e.g. footprintID-pointID')
    parser.add_argument('--box_buffer', default=100, type=float, help='minimum distance between point and border (m)')
    parser.add_argument('--out_file', default='matches.dbf', type=str, help='output file name')
    parser.add_argument('--n_workers', type=int, help='number of workers for multiprocessing pool')
    return parser.parse_args()


def find_overlap(footprint_shp, pol_shp, box_buffer, id_columns, output_name, n_workers):
    """
    Checks if polygon falls inside a raster on footprint shapefile. Distance to border can be used to ensure rasters
    containing points can be cropped to given size.

    :param footprint_shp:
    :param pol_shp:
    :param id_columns:
    :param box_buffer:
    :param output_name:
    :param n_workers:
    :return:
    """
    # get id_columns
    id_columns = id_columns.split('-')

    # read files
    footprint = gpd.read_file(footprint_shp)
    pol = gpd.read_file(pol_shp)

    # convert polygon geometry to bounding box
    bounds = pol['geometry'].bounds
    bboxes = [box(row.minx - box_buffer,
                  row.miny - box_buffer,
                  row.maxx + box_buffer,
                  row.maxy + box_buffer) for _, row in bounds.iterrows()]

    # box_buffer bounding boxes
    old_geo = pol.copy()
    pol = pol.assign(geometry=bboxes)

    # check if point and shapefile are using the same CRS
    if footprint.crs != pol.crs:
        pol = pol.to_crs(footprint.crs)
        old_geo = old_geo.to_crs(footprint.crs)
    # get footprint entries that are most reliable for guano detection (DEC 15 - JAN 15)
    # footprint = footprint[np.array(footprint.dayofseaso > 197) * np.array(footprint.dayofseaso < 229)]

    # matching function
    def match(scene, pol, scene_id, point_id):
        out = {'scene': [], 'site': [], 'geometry': [], 'year': [], 'path': [], 'pol_idx': []}
        for idx, bbox in pol.iterrows():
            try:
                if bbox.geometry.within(scene.geometry):
                    out['scene'].append(scene[scene_id])
                    out['site'].append(bbox[point_id])
                    out['geometry'].append(old_geo.iloc[idx]['geometry'])
                    out['year'].append(scene.year)
                    out['path'].append(scene.location)
                    out['pol_idx'].append(idx)
            except:
                continue
        return out

    # iter through items using multiprocessing pool
    scenes = [scene for _, scene in footprint.iterrows()]
    pool = Pool(n_workers)
    out = pool.map(partial(match, pol=pol, scene_id=id_columns[0], point_id=id_columns[1]), scenes)
    pool.close()
    pool.join()

    # aggregate output
    out_fixed = {'scene': [], 'site': [], 'geometry': [], 'year': [], 'path': [], 'pol_idx': []}
    for ele in out:
        for key in ele:
            out_fixed[key].extend(ele[key])

    # geopandas dataframe to store output in shapefile
    matches = gpd.GeoDataFrame(data=out_fixed, crs=pol.crs)

    # save output to csv
    matches.to_file(output_name)


def main():
    args = parse_args()
    find_overlap(footprint_shp=args.footprint_shp,
                 pol_shp=args.pol_shp,
                 box_buffer=args.box_buffer,
                 id_columns=args.id_columns,
                 output_name=args.out_file,
                 n_workers=args.n_workers)


if __name__ == "__main__":
    main()
