"""
   Script to compare point data and print evaluation metrices
"""

import os
import sys

import argparse
import geopandas as gpd

from shapely.geometry import Point
from tqdm import tqdm

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
    total_nearest_points = 0
    total_distance = 0
    
    for x1, y1 in tqdm(actual_data_list,
                    desc='Processing : ',
                    leave=False,
                    file=sys.stdout):
        dis = min([Point(x1, y1).distance(Point(x2, y2)) for x2, y2 in preicted_data_list])
        if dis < args['threshold']:
            total_nearest_points += 1
            total_distance += dis

    print('Avg distance error:', total_distance/total_nearest_points)
    print('Accuracy:', (total_actual_points/total_nearest_points)*100)
