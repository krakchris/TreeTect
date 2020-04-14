'''
    script to convert tiff files into jpg(RBG) formate
'''

import os

import argparse
import numpy as np
from tifffile import imread

from PIL import Image
from skimage import exposure

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to the csinput directory containing tif files",
                    type=str)
parser.add_argument("--output_dir", help="path to the output directory",
                    type=str)
args = vars(parser.parse_args())

for tif_file in os.listdir(args['input_dir']):
    print('Processing : ', tif_file)

    img = imread(os.path.join(args['input_dir'], tif_file))

    img_rgb = img[:, :, 1:4][:, :, ::-1] # default formate of pillow is RGB

    p1, p99 = np.percentile(img_rgb, (0.8, 99.8))
    img_rgb = exposure.rescale_intensity(img_rgb, in_range=(p1, p99))

    img_rgb = (img_rgb * 255 / np.max(img_rgb)).astype('uint8')
    dst_path = os.path.join(args['output_dir'], tif_file.split('.')[0] + '.jpg')

    im = Image.fromarray(img_rgb)
    im.save(dst_path)

print('process completed successfully')
