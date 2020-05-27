"""
    script to chunk tif files
    input is the tif files directory
    -> Input:
            - Path to the tif directory
            - Path to output directory
    -> command to run:
        python ensemble.py\
            --tif_dir=<PATH TO THE TIF FILES DIRECTORY>\
            --output_dir=<PATH TO THE OUTPUT DIRECTORY>
    -> Output:
        - chunked tif files
"""

import os

import argparse
import numpy as np
import rasterio
from rasterio.mask import mask
import slidingwindow as sw

from shapely.geometry import Polygon
from tqdm import tqdm

def arguments():
    '''
        command line arguments
        retun command line argument dictionary
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif_dir", help="Path to the directory containing tif files",
                        type=str)
    parser.add_argument("--output_dir", help="Path to the output directory",
                        type=str)

    return vars(parser.parse_args())

if __name__ == "__main__":

    CHUNK_SIZE_PIX = 400
    OVERLAP_FRAC = 0.0

    args = arguments()

for big_tif_file_name in tqdm(os.listdir(args['tif_dir']), desc='Processing_tif_files :'):

    big_tif_file_path = os.path.join(args['tif_dir'], big_tif_file_name)
    dataset = rasterio.open(big_tif_file_path)

    # convert bounds to polygon
    x_min_data, y_min_data, x_max_data, y_max_data = dataset.bounds
    poly_raster_bounds = Polygon([(x_min_data, y_min_data),
                                  (x_min_data, y_max_data),
                                  (x_max_data, y_max_data),
                                  (x_max_data, y_min_data)])

    # get raster metadata
    x_pix_size_m = dataset.meta['transform'][0]
    y_pix_size_m = dataset.meta['transform'][4]

    x_raster_size_pix = dataset.meta['width']
    y_raster_size_pix = dataset.meta['height']

    # Generate the set of windows
    windows = sw.generate(np.rot90(np.fliplr(dataset.read().T)),
                          sw.DimOrder.HeightWidthChannel,
                          CHUNK_SIZE_PIX, OVERLAP_FRAC)

    for i in tqdm(range(len(windows)), desc='Chopping tif file : ', leave=False):

        # convert chunk coordinates to bbox
        x_min = (windows[i].x * x_pix_size_m) + x_min_data
        x_max = ((windows[i].x + windows[i].w) * x_pix_size_m) + x_min_data
        y_min = (windows[i].y * abs(y_pix_size_m)) + y_min_data
        y_max = ((windows[i].y + windows[i].h) * abs(y_pix_size_m)) + y_min_data

        x_min_data, y_min_data, x_max_data, y_max_data = dataset.bounds

        # clip raster file
        poly_chunk_bounds = Polygon([(x_min, y_min), (x_min, y_max),
                                     (x_max, y_max), (x_max, y_min)])
        out_img_chunk, out_transform_chunk = mask(dataset,
                                                  shapes=[poly_chunk_bounds],
                                                  crop=True)

        # loop over tree bboxes
        x_min_chunk = out_transform_chunk[2]
        y_min_chunk = out_transform_chunk[5]

        #### tif file #####
        # generate tiff/ profile
        profile = dataset.profile
        profile['transform'] = out_transform_chunk
        profile['width'] = windows[i].w
        profile['height'] = windows[i].h

        # write tif file
        small_tif_file_name = big_tif_file_name.split('.')[0]
        chunked_tif_dir_path = os.path.join(args['output_dir'], small_tif_file_name)

        if not os.path.exists(chunked_tif_dir_path):
            os.makedirs(chunked_tif_dir_path)

        small_tif_file_name = small_tif_file_name + '_{0:03d}.tif'.format(i)
        small_tif_file_path = os.path.join(chunked_tif_dir_path, small_tif_file_name)
        with rasterio.open(small_tif_file_path, 'w', **profile) as dst:
            dst.write(out_img_chunk)
