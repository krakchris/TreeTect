'''
    script to convert tiff files into jpg/png using given bands
    Input:
        - path to the input(tif) directory
        - path to the output directory
        - file_type (jpg/png)
    Output:
        - directory containing images
'''

import os
import sys

import argparse
import numpy as np
import rasterio

from skimage import exposure
from skimage.io import imsave
from tqdm import tqdm


def convert_to_jpg(tif_file_path, band_list):
    '''
        Method to convert tif into jpg/png
        params:
            tif_file_path : path to the tif files
            band_list : list of bands
        return:
            numpy array of image with given bands
    '''
    upper_percentile = 98
    lower_percentile = 2
    max_single_value_count = 600

    dataset = rasterio.open(tif_file_path)
    img = dataset.read()

    if 'ndvi' in band_list:

        if img.shape[0] == 8:
            RED = img[[4], :, :]
            NIR = img[[6], :, :]

        elif img.shape[0] == 4:
            RED = img[[0], :, :]
            NIR = img[[3], :, :]

        else:
            raise Exception('Error: Tif file is not of 4 or 8 bands')

        ndvi = np.where(
            (NIR+RED) == 0.,
            0,
            (NIR-RED)/(NIR+RED))

        ndvi_index_no = band_list.index('ndvi')

        band_list[ndvi_index_no] = 0
        band_list = list(map(int, band_list))

        img_plot_raw = img[band_list, :, :]
        img_plot_raw[[ndvi_index_no], :, :] = ndvi

    else:
        band_list = list(map(int, band_list))
        img_plot_raw = img[band_list, :, :]

    img_plot = np.rot90(np.fliplr(img_plot_raw.T))

    # correct exposure for each band individually
    img_plot_enhance = np.array(img_plot, copy=True)

    if img.shape[0] == 4:
        img_plot = img_plot.astype('float32')

    for band in range(3):
        # check max amount of a single value
        values, counts = np.unique(img_plot, return_counts=True)
        index_nodata = np.argmax(counts)
        nodata_value = values[index_nodata]
        max_count_single_value = np.max(values)

        # if there are more than specific values set them as nan
        if max_count_single_value > max_single_value_count:
            img_plot[img_plot == nodata_value] = np.nan

        p_1, p_2 = np.nanpercentile(img_plot[:, :, band], (lower_percentile, upper_percentile))
        img_plot_enhance[:, :, band] = exposure.rescale_intensity(img_plot[:, :, band],
                                                                  in_range=(p_1, p_2),
                                                                  out_range=(0, 255))

    return img_plot_enhance.astype('uint8')

if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to the input directory containing tif files",
                        type=str)
    parser.add_argument("--output_dir", help="path to the output directory",
                        type=str)
    parser.add_argument("--file_type", help="output file type (jpg/png)",
                        type=str, default='jpg')
    args = vars(parser.parse_args())

    is_continue = True

    while is_continue:

        band = input(f'Enter band no. seperated by comma or z/Z to break: \n').split(', ')

        if band[0] in ['z', 'Z']:
            break

        output_dir_path = os.path.join(args['output_dir'], '_'.join(band))

        # creating output directory if not exists
        if not os.path.exists(output_dir_path):
            print(f'creating folder:{output_dir_path}')
            os.makedirs(output_dir_path)

        else:
            print(f'folder already present, data may get override')
            if input('Do you want to contiue y/Y: ') not in ['y' or 'Y']:
                break

        # processing each tif files
        for tif_file in tqdm(os.listdir(args['input_dir']), file=sys.stdout):

            if not tif_file.endswith(('.tif')):
                continue

            img = convert_to_jpg(
                os.path.join(args['input_dir'], tif_file),
                band.copy())

            dst_path = os.path.join(
                output_dir_path,
                tif_file.split('.')[0] + f".{args['file_type']}")

            imsave(dst_path, img)
