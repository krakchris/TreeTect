"""
   Script to compare point data and print evaluation metrices
"""

import os
import sys

import argparse
import geopandas as gpd

from shapely.geometry import Point
from tqdm import tqdm

from multiprocessing import Pool, Value
from p_tqdm import p_map

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
                        help="Threshold float value, distance less than threshold will be considered",
                        type=float)

    return vars(parser.parse_args())

def find_minimum_distance(point):

    x1, y1 = point
    global total_nearest_points
    global total_distance

    dis = min([Point(x1, y1).distance(Point(x2, y2)) for x2, y2 in preicted_data_list])
    if dis < args['threshold']:
        with total_nearest_points.get_lock():
            total_nearest_points.value += 1
        with total_distance.get_lock():
            total_distance.value += dis
    
if __name__ == "__main__":
    args = arguments()

    # load data
    print('Loading df...') 
    actual_df = gpd.read_file(args['actual_shp_file'])
    preicted_df = gpd.read_file(args['predicted_shp_file'])

    # remove duplicates
    print('Removing duplicates...')
    actual_data_list = list(set([(point.x, point.y) for point in actual_df['geometry']]))
    preicted_data_list = list(set([(point.x, point.y) for point in preicted_df['geometry']]))

    # calculate minimum distance
    total_actual_points = len(actual_data_list)
    total_nearest_points = Value('i', 0)
    total_distance = Value('f', 0)

    p_map(find_minimum_distance, actual_data_list[:20])
    
    print('Avg distance error:', total_distance.value/total_nearest_points.value)
    print('Accuracy:', (total_nearest_points.value/total_actual_points)*100)
