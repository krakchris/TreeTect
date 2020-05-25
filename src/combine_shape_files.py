"""
    -> Script to combine small shape files into bigger one.
    -> Input:
            - Path to small shape files directory
            - Path to output directory
    -> command to run:
        python combine_shape_files.py\
            --input_dir=<PATH_TO_THE_DIRECTORY_CONTAINING_SHAPE_FILES>\
            --output_dir=<PATH TO THE OUTPUT DIRECTORY>
    -> Output:
        - big size shape files
"""

import glob
import os

import argparse
import geopandas as gpd
import pandas as pd

from tqdm import tqdm

def arguments():
    '''
        command lines arguments
        return dictionary of command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Path to the directory containing shape files",
                        type=str)
    parser.add_argument("--output_dir", help="Path to the output directory",
                        type=str)
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = arguments()

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    # listing all shape files recursively in the folder
    shape_file_path_list = glob.glob(args['input_dir'] + '/**/*.shp', recursive=True)

    dst_file_name = '_'.join(shape_file_path_list[0].strip().split('/')[-1].split('_')[:-2])

    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(shape_file_path)
                                      for shape_file_path in tqdm(shape_file_path_list,
                                                                  desc='processing _shape files',
                                                                  file=sys.stdout)],
                                     ignore_index=True),
                           crs=gpd.read_file(shape_file_path_list[0]).crs)

    gdf.to_file(os.path.join(args['output_dir'], dst_file_name),
                driver='ESRI Shapefile')
