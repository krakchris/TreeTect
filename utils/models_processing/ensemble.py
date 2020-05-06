"""
    Note : first three words seperated by underscore in model file name denotes bands
           e.g: 1_ndvi_3_frozen_inference_graph.pb

    -> Model Ensembling script ensemble different model's output.
    -> Input:
            - Path to frozen's model's directory
            - Path to tif image directory
            - Path to output directory
            - Path to label file
            - threeshold value
    -> command to run:
        python ensemble.py\
            --model_dir=<PATH_TO_THE_DIRECTORY_CONTAINING_DIFFERENT_MODEL>\
            --output_dir=<PATH TO THE OUTPUT DIRECTORY>\
            --input_dir=<PATH_TO_THE_DIRECTORY_CONTAINING_TIF_FILES>\
            --label_file=<PATH TO THE LABEL FILE>\
            --threshold=<Threshold value for inference default is 0.5>
    -> Output:
        - Image file having rectangles drawn on it
"""
import math
import os
import sys

from collections import defaultdict

import csv
import fiona
import rasterio

from PIL import Image, ImageDraw
from shapely.geometry import Polygon, mapping, box
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..") # for sibling import
from  model_test import *
from data_processing.convert_tiff_into_jpeg import convert_to_jpg

def get_inference_data(args):
    '''
        Method to run inference for each tif on all model
        and return the compiledinference data

        params:
            args : commandline argument's dictionary

        return a dictionary where
        key  = tif_file_name
        value = [[[boundary_box_1], class, score, model_name(of this inference)],
                  [[boundary_box_2], class, score, model_name(of this inference)],
                  ...   ]
    '''

    category_index = label_map_util.create_category_index_from_labelmap(args['label_file'],
                                                                        use_display_name=True)
    tif_inference_data = defaultdict(list)

    # looping over all model in the directory
    for model_file_name in tqdm(os.listdir(args['model_dir']), desc='Model_files'):

        model_file_path = os.path.join(args['model_dir'], model_file_name)
        detection_graph = get_detection_graph(model_file_path)
        band_list = model_file_name.split('_')[:3]

        # Processing tif files
        for tif_file_name in tqdm(os.listdir(args['input_dir']), desc='tif_files', leave=False):

            tif_file_path = os.path.join(args['input_dir'], tif_file_name)
            img_np = convert_to_jpg(
                                tif_file_path,
                                band_list.copy())

            height, width, _ = img_np.shape

            output_dict = run_inference_for_single_image(img_np, detection_graph)

            for index, detection_score in enumerate(output_dict['detection_scores']):
                if detection_score >= args['threshold']:
                    ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]

                    tif_inference_data[tif_file_name].append([
                        list(map(int, [xmin*width, ymin*height, xmax*width, ymax*height])),
                        output_dict['detection_classes'][index],
                        output_dict['detection_scores'][index],
                        model_file_name])

    return tif_inference_data

def is_overlapping(box1, box2):

    overlapping_threshold = 0.20

    xmin, ymin, xmax, ymax = box1
    poly1 = Polygon([
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymax),
        (xmax, ymin)])

    xmin, ymin, xmax, ymax = box2
    poly2 = Polygon([
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymax),
        (xmax, ymin)])

    intersection = poly1.intersection(poly2)
    union = poly1.union(poly2)

    if (float(intersection.area)/union.area) > overlapping_threshold:
        return True
    return False

def optimize_bounding_boxes(tif_inference_data):
    '''
        Method to remove the overlapping boundary boxes
        params:
            tif_inference_data

        return optimized_tif_inference_data
    '''

    optimized_tif_inference_data = defaultdict(list)

    # processing each tif's data
    for tif_file_name in tqdm(tif_inference_data.keys(), desc='optimizing'):
        for i in range(len(tif_inference_data[tif_file_name])):

            if tif_inference_data[tif_file_name][i][-1] == 'p':
                continue   # skipping overlapped data which is already checked

            temp_list = []

            for j in range(i+1, len(tif_inference_data[tif_file_name])):

                if tif_inference_data[tif_file_name][j][-1] == 'p':
                    continue  # skipping overlapped data which is already checked

                box_1 = tif_inference_data[tif_file_name][i][0]
                class1 = tif_inference_data[tif_file_name][i][1]
                box_2 = tif_inference_data[tif_file_name][j][0]
                class2 = tif_inference_data[tif_file_name][j][1]

                if is_overlapping(box_1, box_2) and class1 == class2:
                    temp_list.append(tif_inference_data[tif_file_name][j])
                    tif_inference_data[tif_file_name][j].append('p')

            temp_list.append(tif_inference_data[tif_file_name][i])

            avg_xmin, avg_ymin, avg_xmax, avg_ymax, avg_score = 0, 0, 0, 0, 0

            for data in temp_list:
                avg_xmin += data[0][0]
                avg_ymin += data[0][1]
                avg_xmax += data[0][2]
                avg_ymax += data[0][3]
                avg_score += data[2]

            avg_xmin = int(avg_xmin/len(temp_list))
            avg_ymin = int(avg_ymin/len(temp_list))
            avg_xmax = int(avg_xmax/len(temp_list))
            avg_ymax = int(avg_ymax/len(temp_list))
            avg_score = int(avg_score)/len(temp_list)

            optimized_tif_inference_data[tif_file_name].append([
                [avg_xmin, avg_ymin, avg_xmax, avg_ymax],
                temp_list[0][1],
                avg_score])

    return optimized_tif_inference_data

def draw_boundary_boxes(optimized_tif_inference_data, args):
    '''
        Method to draw optimized boundary boxes over images and save to output_directory
        params:
            optimized_tif_inference_data
            args : command line arguments dictionary
    '''
    dst_path = os.path.join(args['output_dir'], 'visualizations')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for tif_file_name in tqdm(optimized_tif_inference_data.keys(), desc='visualization'):
        img_np = convert_to_jpg(
            os.path.join(args['input_dir'], tif_file_name),
            [4, 3, 2])

        img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img)

        for box in optimized_tif_inference_data[tif_file_name]:
            xmin, ymin, xmax, ymax = box[0]
            draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=None, width=2, outline=(0, 255, 0))

        img.save(os.path.join(dst_path, tif_file_name.split('.')[0] + '.jpg'))

def generate_shape_files(optimized_tif_inference_data, args):
    '''
        Method to create shpfiles and save to output_directory
        params:
            optimized_tif_inference_data
            args : command line arguments dictionary
    '''
    dst_path = os.path.join(args['output_dir'], 'inference_shape_files')

    m2ftconversion = 3.28084

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for tif_file_name in tqdm(optimized_tif_inference_data.keys(), desc='shape_files'):
        tif_file_path = os.path.join(args['input_dir'], tif_file_name)

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
            'geometry': 'Polygon',
            'properties': {'score': 'float',
                           'ns_spread' : 'float',
                           'ew_spread' : 'float',
                           'volume' : 'float',
                           'ndvi_avg' : 'float',
                           'savi_avg' : 'float',
                           'evi_avg': 'float'},
            }

        # Write a new Shapefile
        with fiona.open(os.path.join(dst_path, tif_file_name.split('.')[0]), 'w',
                        crs=crs,
                        driver='ESRI Shapefile',
                        schema=schema) as c:

            for predicted_data in optimized_tif_inference_data[tif_file_name]:

                xmin, ymin, xmax, ymax = predicted_data[0]

                crown_image = image_array[:, ymin:ymax, xmin:xmax]

                RED = crown_image[0, :, :].astype(np.float32)
                GREEN = crown_image[1, :, :].astype(np.float32)
                BLUE = crown_image[2, :, :].astype(np.float32)
                NIR = crown_image[3, :, :].astype(np.float32)

                ## vegetation indices
                # NDVI
                ndvi = (NIR - RED) / (NIR + RED)
                ndvi_avg = np.average(ndvi)

                # EVI
                G = 2.5; L = 2.4; C = 1
                evi = G*((NIR-RED)/(L+NIR+C*RED))
                evi_avg = np.average(evi)

                # SAVI
                L = 0.5
                savi = ((NIR - RED) / (RED + NIR + L)) * (1+L)
                savi_avg = np.average(savi)

                # calculate spread of crown
                north_south_spread = ((ymax - ymin) * y_res) * m2ftconversion
                east_west_spread = ((xmax - xmin) * x_res) * m2ftconversion
                ns_spread = north_south_spread
                ew_spread = east_west_spread

                # calculate area
                area = north_south_spread * east_west_spread

                # calculate volume
                volume = (4/3
                          * math.pi
                          * north_south_spread
                          * east_west_spread
                          * (((north_south_spread+east_west_spread)/2)/2))

                ndvi[0, :] = 0
                ndvi[-1, :] = 0
                ndvi[:, 0] = 0
                ndvi[:, -1] = 0

                ndvi = gaussian_filter(ndvi, sigma=2)

                # recalculate coordinates
                xmin = (xmin * x_res + (dataset.bounds.left))
                xmax = (xmax * x_res + (dataset.bounds.left))

                ymax = (raster_size_y - (ymax * y_res)) + dataset.bounds.bottom
                ymin = (raster_size_y - (ymin * y_res)) + dataset.bounds.bottom

                poly = box(xmin, ymax, xmax, ymin)

                c.write({
                    'geometry': mapping(poly),
                    'properties': {'score': float(predicted_data[2]),
                                   'ns_spread': float(ns_spread),
                                   'ew_spread': float(ew_spread),
                                   'volume': float(volume),
                                   'ndvi_avg': float(ndvi_avg),
                                   'savi_avg': float(savi_avg),
                                   'evi_avg' : float(evi_avg)}
                })

def generate_csv(optimized_tif_inference_data, args):
    '''
        Method to log bounding box data in a CSV file
        params:
            optimized_tif_inference_data
            args : command line arguments dictionary
    '''
    csv_file_path = os.path.join(args['output_dir'], 'annotations.csv')

    with open(csv_file_path, 'w') as csv_file:
        writer_obj = csv.writer(csv_file)
        writer_obj.writerow(['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score'])

        for tif_file_name in tqdm(optimized_tif_inference_data.keys(), desc='csv_file'):
            for data in optimized_tif_inference_data[tif_file_name]:
                writer_obj.writerow([
                    tif_file_name,
                    data[0][0],
                    data[0][1],
                    data[0][2],
                    data[0][3],
                    data[1],
                    round(data[2], 2)])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Path to the directory containing various models",
                        type=str)
    parser.add_argument("--input_dir", help="Path to the directory containing tif files",
                        type=str)
    parser.add_argument("--output_dir", help="Path to the output directory",
                        type=str)
    parser.add_argument("--label_file", help="Path to the label file",
                        type=str)
    parser.add_argument("--threshold", help="Threshold value for inference",
                        type=float, default=0.5)

    args = vars(parser.parse_args())

    tif_inference_data = get_inference_data(args)

    print('oprtmizing inference results...')
    optimized_tif_inference_data = optimize_bounding_boxes(tif_inference_data)

    print('generating visualizations...')
    draw_boundary_boxes(optimized_tif_inference_data, args)

    print('generating shape files...')
    generate_shape_files(optimized_tif_inference_data, args)

    print('generating csv file...')
    generate_csv(optimized_tif_inference_data, args)
