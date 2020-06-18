"""
   Script to compare point data and generate evaluation metrices
   -> Input:
            - Path to actual data shape file

    -> command to run:
        python point_data_comparison.py\
            --actual_shp_file=<PATH_TO_THE_ACTUAL_SHAPE_FILE>\
            --predicted_shp_file=<PATH TO THE PREDICTED SHAPE FILE>\
            --threshold=<RADIUS_OF_AREA_IN WHICH_TREES_WILL_BE_SEARCHED>
            --output_dir=<PATH TO THE OUTPUT DIRECTORY>\
    -> Output:
        - Line shape files from actual to predicted point for the nearest tree
"""

import math
import os
import sys
import shutil

import argparse
import fiona
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, box, mapping
from tqdm import tqdm
from scipy.spatial.distance import cdist

# constants
CRS = 'EPSG:32631'
TEMP_DIR_PATH = '../temp'
ACTUAL_SHAPE_FILE_LOCAL_PATH = os.path.join(TEMP_DIR_PATH, 'actual.shp')
PREDICTED_SHAPE_FILE_LOCAL_PATH = os.path.join(TEMP_DIR_PATH, 'predicted.shp')
GRAPH_RESCALE_FACTOR = 1000


def arguments():
    '''
        command line arguments
        retun command line argument dictionary
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual_shp_file",
                        help="Path to actual point data shape file",
                        type=str)
    parser.add_argument("--predicted_shp_file",
                        help="Path to the predicted point data shape file",
                        type=str)
    parser.add_argument("--threshold",
                        help="Threshold float value, trees within this range will be considered",
                        type=float)
    parser.add_argument("--output_dir",
                        help="Path to the output_dir",
                        type=str)

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = arguments()

    if os.path.exists(TEMP_DIR_PATH):
        shutil.rmtree(TEMP_DIR_PATH)

    os.makedirs(TEMP_DIR_PATH)

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    # load data
    print('Loading df...')
    actual_df = gpd.read_file(args['actual_shp_file']).to_crs(CRS)
    predicted_df = gpd.read_file(args['predicted_shp_file']).to_crs(CRS)

    # save the df and reopen with fiona to find the extent
    actual_df.to_file(ACTUAL_SHAPE_FILE_LOCAL_PATH, driver='ESRI Shapefile')
    predicted_df.to_file(PREDICTED_SHAPE_FILE_LOCAL_PATH, driver='ESRI Shapefile')

    print('Extracting extent information...')
    with fiona.open(ACTUAL_SHAPE_FILE_LOCAL_PATH, 'r') as shp_file:
        actual_extent = shp_file.bounds
    with fiona.open(PREDICTED_SHAPE_FILE_LOCAL_PATH, 'r') as shp_file:
        pedicted_extent = shp_file.bounds

    # find the intersection area of two shape files
    actual_polygon_obj = box(actual_extent[0],
                             actual_extent[1],
                             actual_extent[2],
                             actual_extent[3])

    predicted_polygon_obj = box(pedicted_extent[0],
                                pedicted_extent[1],
                                pedicted_extent[2],
                                pedicted_extent[3])

    common_area = actual_polygon_obj.intersection(predicted_polygon_obj)
    x_coor, y_coor = common_area.exterior.coords.xy

    # load geopandas df again only for common area
    #  adding margin of 3 meters
    print('Extracting common area data...')
    actual_df = gpd.read_file(ACTUAL_SHAPE_FILE_LOCAL_PATH,
                              bbox=(min(x_coor), min(y_coor), max(x_coor), max(y_coor)))

    predicted_df = gpd.read_file(PREDICTED_SHAPE_FILE_LOCAL_PATH,
                                 bbox=(min(x_coor) - 3, min(y_coor) - 3, max(x_coor) + 3, max(y_coor) + 3))

    # remove duplicates and load points data in a list
    print('Removing duplicates...')
    actual_data_list = actual_df.drop_duplicates('geometry')['geometry'].to_list()
    predicted_data_list = predicted_df.drop_duplicates('geometry')['geometry'].to_list()

    actual_data_np_array = np.array(list(map(lambda point: (point.x, point.y), actual_data_list)))
    predicted_data_np_array = np.array(list(map(lambda point: (point.x, point.y),
                                                predicted_data_list)))

    # decalre variables
    total_dis_bw_two_pts = 0.0
    result_line_obj_list = []
    total_predicted_points = len(predicted_data_np_array)
    x_coor_list = []
    y_coor_list = []
    quad_1 = 0
    quad_2 = 0
    quad_3 = 0
    quad_4 = 0

    for actual_point in tqdm(actual_data_np_array, desc='Processing', file=sys.stdout):

        # calculate distance from actual point to all predicted points
        dis_array = cdist([actual_point], predicted_data_np_array)[0]

        min_dis = min(dis_array)

        if min_dis <= args['threshold']:
            total_dis_bw_two_pts += min_dis
            min_dis_index = np.where(dis_array == min_dis)[0][0]
            predicted_point = tuple(predicted_data_np_array[min_dis_index])

            result_line_obj_list.append(LineString([actual_point,
                                                    predicted_point]))
            # storing x, y points in a list to draw graphs
            x = (predicted_point[0] - actual_point[0]) * GRAPH_RESCALE_FACTOR
            y = (predicted_point[1] - actual_point[1]) * GRAPH_RESCALE_FACTOR
            x_coor_list.append(x)
            y_coor_list.append(y)

            if x > 0  and y > 0:
                quad_1 += 1
            elif x < 0 and y > 0:
                quad_2 += 1
            elif x < 0 and y < 0:
                quad_3 += 1
            elif x > 0 and y < 0:
                quad_4 += 1 

            predicted_data_np_array = np.delete(predicted_data_np_array, min_dis_index, axis=0)

    print('Generating result.txt...')
    result_file_path = os.path.join(args['output_dir'], 'result.txt')

    with open(result_file_path, 'w') as res_file_obj:
        res_file_obj.write(f'Total actual points : {len(actual_data_np_array)}\n')
        res_file_obj.write(f'Total predicted points : {total_predicted_points}\n')
        res_file_obj.write(f'Total predicted points that are not near to any actual point : {len(predicted_data_np_array)}\n')
        res_file_obj.write(f'Total actual points : {len(result_line_obj_list)}\n')
        res_file_obj.write(f'Average distance : {total_dis_bw_two_pts/len(result_line_obj_list)}\n')
        res_file_obj.write(f'Accuracy : {(len(result_line_obj_list)/ float(len(actual_data_np_array)))*100}\n')
        res_file_obj.write(f'Total points in quadrant 1 : {quad_1}\n')
        res_file_obj.write(f'Total points in quadrant 2 : {quad_2}\n')
        res_file_obj.write(f'Total points in quadrant 3 : {quad_3}\n')
        res_file_obj.write(f'Total points in quadrant 4 : {quad_4}\n')
        res_file_obj.write(f'Resultant angle : {math.degrees(math.atan(sum(x_coor_list)/sum(y_coor_list)))}\n')
        res_file_obj.write(f'Resultant magnitude : {math.sqrt(sum(x_coor_list)**2 + sum(y_coor_list)**2)/GRAPH_RESCALE_FACTOR}')

    print('Creating shape file...')
    schema = {
        'geometry': 'LineString',
        'properties': {}
        }

    with fiona.open(os.path.join(args['output_dir'],
                                 'result_line_comparison'), 'w',
                    crs=CRS,
                    driver='ESRI Shapefile',
                    schema=schema) as c:
        for line_obj in result_line_obj_list:
            c.write({
                'geometry': mapping(line_obj),
                'properties': {}})

    print('Generation plot image...')
    plt.figure(figsize=(20,20))
    plt.plot([[0]*len(x_coor_list), x_coor_list],
            [[0]*len(y_coor_list), y_coor_list]) 
    plt.savefig(os.path.join(args['output_dir'], 'res.png'))

    shutil.rmtree(TEMP_DIR_PATH)
