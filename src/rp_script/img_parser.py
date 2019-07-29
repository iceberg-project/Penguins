# pylint: disable=protected-access, unused-argument
# pylint: disable=no-value-for-parameter
"""Penguin Image locater and parser"""
import glob
import json
import sys
import argparse

def img_parser():
    """
    Traversing over all the files inside the target
    dataset path and extract each image information
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path of the dataset')
    json_dict = {}
    data = []
    for filename in glob.iglob(sys.argv[1] + '*.png'):
        print 'found a PNG Images ' + filename
        tmp_dict = {}
        tmp_dict["img"] = filename
        data.append(tmp_dict)

    json_dict["Dataset"] = data

# Writing information to JSON file
    with open("penguins_images.json", "w") as outfile:
        json.dump(json_dict, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    img_parser()
