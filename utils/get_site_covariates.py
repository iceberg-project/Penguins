import argparse
import os
import re

import pandas as pd
import rasterio

__all__ = ['get_covariate_data']


# argument parser
def parse_args():
    parser = argparse.ArgumentParser(prog='covariate scrapping from MAPPPD')
    parser.add_argument('--scenes_folder', help='path to folder with input images')
    parser.add_argument('--covariates_folder', help='path to folder with raw MAPPPD covariates')
    parser.add_argument('--output_file', default='input_covariates.csv', help='path to output file')
    return parser.parse_args()


# helper function to get covariates from scene
def get_covariate_data(input_scene, covariate_folder):
    """
    Gets covariate data from MAPPPD for a given scene. Requires a image to site mapping reference table and covariate
    data on selected sites from MAPPPD (http://www.penguinmap.com/mapppd).

    :param input_scene:
    :param covariate_folder:
    :return: covariates pd.DataFrame
    """
    # data frame to store covariates
    covariates = pd.DataFrame()

    # extract pixel width, height from input image using rasterio and add to covariates
    if input_scene.split('.')[-1] == 'tif':
        with rasterio.open(input_scene) as img:
            covariates['pixel_width'] = [img.transforms[0]]
            covariates['pixel_height'] = [img.transforms[4]]

    # get rid of path on input_scene
    input_scene = os.path.basename(input_scene)

    # get sensor and time data parsing input image name and add to covariates
    sensor = re.search(r'[A-Z]{2}\d{2}', input_scene).group(0)
    time_stamp = re.search(r'\d{2}[A-Z]{3}\d{6}', input_scene).group(0)
    year = '20' + time_stamp[:2]
    month = time_stamp[2:5]
    season = int(year) - 1 if int(month) <= 5 else int(year)
    month_offset = [1, -1, -1, 0, 0, 1, 2, 3, 3, 4, 4, 5]
    julian_day = int(month) * 30 + month_offset[int(month) - 1] + (int(year) % 4 == 0) + time_stamp[5:7]
    hour = time_stamp[7:9]

    # add covariates
    covariates['sensor'] = [sensor]
    covariates['year'] = [year]
    covariates['season'] = [season]
    covariates['julian_day'] = [julian_day]
    covariates['hour'] = [hour]

    # find site
    scene_to_site = pd.read_csv(f"{covariate_folder}/scene_to_site.csv")
    scene_to_site["scene"] = [ele.split('.')[0] for ele in scene_to_site['scene']]
    site = scene_to_site.loc[scene_to_site.scene == input_scene, 'site'].values[0]

    # get model and location covariates for that site
    site_covariates = pd.read_csv(f"{covariate_folder}/selected_sites_V_2.1.csv")
    site_covariates = site_covariates.loc[site_covariates['site name'] == site, site_covariates.columns[1:]]

    for col, val in site_covariates.iteritems():
        covariates[col] = [val.values[0]]

    # get model predictions for that site for that year and a few previous years
    model_covariates = pd.read_csv(f"{covariate_folder}/Model_results_by_site_V_2.1.csv")
    covariates_site = model_covariates.loc[model_covariates['site name'] == site]
    for idx, szn in enumerate(range(int(season), int(season) - 5, -1)):
        covariates_site_season = covariates_site.loc[covariates_site['season starting'] == szn]
        for metric in ['lower CI', 'mean', 'upper CI']:
            for spcs in ['adelie penguin', 'chinstrap penguin', 'gentoo penguin']:
                if spcs in list(covariates_site_season['common name']):
                    covariates[spcs] = [1]
                    covariates[f"{idx}_{spcs}_{metric}"] = \
                        covariates_site_season.loc[covariates_site_season['common name'] == spcs, metric].values[0]
                else:
                    covariates[spcs] = [0]
                    covariates[f"{idx}_{spcs}_{metric}"] = [0]

    # substitute spaces in column headers for '_'
    covariates.columns = [col.replace(' ', '_') for col in covariates.columns]

    return covariates


def main():
    args = parse_args()
    covariates_all = pd.DataFrame()

    # get covariates for all scenes in folder
    for scn in os.listdir(args.scenes_folder):
        covariates_all = covariates_all.append(get_covariate_data(scn, args.covariates_folder))

    # save to output .csv
    covariates_all.to_csv(args.output_file)


if __name__ == "__main__":
    main()
