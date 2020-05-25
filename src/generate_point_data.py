"""
    -> Script to generate point data i.e tree location shape files
    -> Input:
            - Path to tif image directory
            - Path to annotation csv file
            - Path to output directory
    -> command to run:
        python ensemble.py\
            --input_dir=<PATH_TO_THE_DIRECTORY_CONTAINING_TIF_FILES>\
            --csv_file=<PATH TO THE ANNOTATION'S CSV FILE>\
            --output_dir=<PATH TO THE OUTPUT DIRECTORY>\
    -> Output:
        - point data shape files for each tif
"""

# importing
import os
import random
import sys

import argparse
import fiona
import numpy as np
import pandas as pd
import rasterio

from scipy import ndimage
from shapely.geometry import Point, mapping
from tqdm import tqdm

def arguments():
    '''
        command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Path to the directory containing tif files",
                        type=str)
    parser.add_argument("--csv_file", help="Path to the label file",
                        type=str)
    parser.add_argument("--output_dir", help="Path to the output directory",
                        type=str)

    return vars(parser.parse_args())

def generate_point_shape_files(tif_file_path, annotations_df, output_dir):
    '''
        method to generate point data shape files tif files
        params:
            tif_file_path : path to the tif file
            annotations_df : pandas data frame of annotations.csv
            output_dir: path to the directory where shape files will get saved
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = rasterio.open(tif_file_path)

    crs = dataset.read_crs()
    image_array = dataset.read()

    # get raster size in meters
    raster_size_x = dataset.bounds.right - dataset.bounds.left
    raster_size_y = dataset.bounds.top - dataset.bounds.bottom

    # get raster resolution in meters
    y_res = abs(dataset.read_transform()[1])
    x_res = abs(dataset.read_transform()[5])

    schema = {
        'geometry': 'Point',
        'properties': {'score': 'float',
                       'area' : 'float',
                       'ndvi_avg' : 'float',
                       'savi_avg' : 'float',
                       'evi_avg': 'float'},
        }

    # Write a new Shapefile
    with fiona.open(os.path.join(output_dir, tif_file_name.split('.')[0]), 'w',
                    crs=crs,
                    driver='ESRI Shapefile',
                    schema=schema) as c:

        # iterate over all annotations for this tif file
        for _, row in annotations_df.iterrows():

            # slice array to tree
            crown_image = image_array[:,
                                      int(row['ymin']):int(row['ymax']),
                                      int(row['xmin']):int(row['xmax'])]

            # get bands
            RED = crown_image[4, :, :].astype(np.float32)
            GREEN = crown_image[2, :, :].astype(np.float32)
            BLUE = crown_image[1, :, :].astype(np.float32)
            NIR = crown_image[7, :, :].astype(np.float32)

            ## vegetation indices
            # NDVI
            ndvi = np.where(
                (NIR+RED) == 0.,
                0,
                (NIR-RED)/(NIR+RED))
            ndvi_avg = np.average(ndvi)

            # EVI
            G = 2.5; L = 2.4; C = 1
            evi = np.where(
                (L+NIR+C*RED) == 0.,
                0,
                G*((NIR-RED)/(L+NIR+C*RED)))
            evi_avg = np.average(evi)

            # SAVI
            L = 0.5
            savi = np.where(
                (RED + NIR + L) == 0.,
                0,
                ((NIR - RED) / (RED + NIR + L)) * (1+L))
            savi_avg = np.average(savi)

            # remove edge pixels
            ndvi[0, :] = 0
            ndvi[-1, :] = 0
            ndvi[:, 0] = 0
            ndvi[:, -1] = 0

            # appy gaussian
            ndvi = ndimage.filters.gaussian_filter(ndvi, sigma=2)

            # apply mask
            ndvi_mask = ndvi > ndvi_avg * 0.75

            area = len(ndvi[ndvi_mask]) * x_res # assuming x_res == y_res

            if area == 0:
                continue

            x, y = map(int, ndimage.measurements.center_of_mass(ndvi_mask))

            # recalculate coordinates
            x = ((row['xmin'] + x) * x_res + (dataset.bounds.left))
            y = (raster_size_y - ((y + row['ymin']) * y_res)) + dataset.bounds.bottom

            pt = Point(x, y)

            c.write({
                'geometry': mapping(pt),
                'properties': {'score': float(row['score']),
                               'area' : float(area),
                               'ndvi_avg': float(ndvi_avg),
                               'savi_avg': float(savi_avg),
                               'evi_avg' : float(evi_avg)}})

if __name__ == '__main__':
    args = arguments()
    all_annotations_df = pd.read_csv(args['csv_file'])

    for tif_file_name in tqdm(os.listdir(args['input_dir']), desc='shape_files', file=sys.stdout):

        if not tif_file_name.endswith(('.tif',)):
            continue

        tif_file_path = os.path.join(
            args['input_dir'],
            tif_file_name)

        # filter annotataions of this particular tif file
        annotations_df = all_annotations_df[all_annotations_df.filename == tif_file_name]

        generate_point_shape_files(
            tif_file_path,
            annotations_df,
            args['output_dir'])
