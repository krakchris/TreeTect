"""
    script to generate training data
    input is the tif files and annotated shape files for each
    Note : shape file should contains same name as tif file name
    -> Input:
            - Path to the tif directory
            - Path to the shape file directory which contains shape files for each tif
            - Path to output directory
    -> command to run:
        python ensemble.py\
            --tif_dir=<PATH_TO_THE_DIRECTORY_CONTAINING_TIF_FILES>\
            --shape_dir=<PATH TO THE SHAPE FILE DIR>\
            --output_dir=<PATH TO THE OUTPUT DIRECTORY>
    -> Output:
        - chunked tif files and annotations.txt
"""

import os
import glob

import argparse
import fiona
import numpy as np
import rasterio
from rasterio.mask import mask
import slidingwindow as sw

from shapely.geometry import shape, Polygon, box, MultiPolygon
from tqdm import tqdm

def arguments():
    '''
        command line arguments
        retun command line argument dictionary
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif_dir", help="Path to the directory containing tif files",
                        type=str)
    parser.add_argument("--shape_dir", help="Path to the dirctory containing shape file",
                        type=str)
    parser.add_argument("--output_dir", help="Path to the output directory",
                        type=str)

    return vars(parser.parse_args())

if __name__ == "__main__":

    # constants
    CHUNK_SIZE_PIX = 400
    OVERLAP_FRAC = 0.0
    LABEL = 'tree'

    args = arguments()

    for tif_file_name in tqdm(os.listdir(args['tif_dir']), desc='processing tif files : '):

        if not tif_file_name.endswith(('.tif',)):
            continue

        tif_file_path = os.path.join(args['tif_dir'], tif_file_name)
        shape_file_path = glob.glob(os.path.join(args['shape_dir'],
                                                 f"**/{tif_file_name.split('.')[0]}.shp"),
                                    recursive=True)[0]

        dataset = rasterio.open(tif_file_path)

        # convert list to shapely MultiPolgyons
        annotations_MultiPoly = MultiPolygon([shape(pol['geometry'])
                                              for pol in fiona.open(shape_file_path)
                                              if pol['geometry'] is not None])

        # crop raster file to annotation extend
        x_min_annotations, y_min_annotations, x_max_annotations, y_max_annotations = annotations_MultiPoly.bounds
        chunk_bbox_org = box(x_min_annotations, y_min_annotations, x_max_annotations, y_max_annotations)
        # crop
        out_img, out_transform = mask(dataset, shapes=[chunk_bbox_org], crop=True)

        ## get raster bbox
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

        for i in tqdm(range(len(windows)), desc='chopping tif files : ', leave=False):

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

            # select only bboxes that are fully within the chunk
            chunk_MultiPoly = MultiPolygon([poly_chunk_bounds.intersection(poly)
                                            for poly in annotations_MultiPoly
                                            if poly.within(poly_chunk_bounds)])

            # only continue if there are boxes
            if len(chunk_MultiPoly) == 0:
                continue

            #### tif file #####
            # generate tiff/ profile
            profile = dataset.profile
            profile['transform'] = out_transform_chunk
            profile['width'] = windows[i].w
            profile['height'] = windows[i].h

            # write tif file
            file_name_tif = tif_file_name.split('.')[0]
            file_name_tif = file_name_tif + '_{0:03d}.tif'.format(i)
            file_path_tif = os.path.join(args['output_dir'], file_name_tif)
            with rasterio.open(file_path_tif, 'w', **profile) as dst:
                dst.write(out_img_chunk)

            # get annotation coordinates
            x = (abs(np.array([poly.exterior.coords.xy[0]
                               for poly in chunk_MultiPoly])
                     - x_min_chunk)
                 * 1/out_transform_chunk[0])

            y = (abs(np.array([poly.exterior.coords.xy[1]
                               for poly in chunk_MultiPoly])
                     - y_min_chunk)
                 * 1/out_transform_chunk[0])

            # write annotations to annotation file
            for i in range(len(x)):
                x_min_bbox = np.round(x[i]).astype(int).min()
                x_max_bbox = np.round(x[i]).astype(int).max()
                y_min_bbox = np.round(y[i]).astype(int).min()
                y_max_bbox = np.round(y[i]).astype(int).max()

                string_list_tif = [file_name_tif,
                                   str(x_min_bbox),
                                   str(y_min_bbox),
                                   str(x_max_bbox),
                                   str(y_max_bbox),
                                   LABEL]

                string_tif = ",".join(string_list_tif)

                with open(os.path.join(args['output_dir'], 'annotations.txt'), 'a') as file:
                    file.write(string_tif + "\n")
                    