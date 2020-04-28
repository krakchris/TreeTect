"""
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

import numpy as np
import rasterio

from pprint import pprint
from skimage import exposure
from skimage.io import imsave, imread

from  model_test import *

def tif_to_image_conversion(tif_file_path, band_list):
    file_type = 'jpg' # jpg or png
    upper_percentile = 98
    lower_percentile = 2
    max_single_value_count = -999

    dataset = rasterio.open(tif_file_path)

     # read and reformat raster data
    img = dataset.read()
    img_plot_raw = img[band_list,:,:]
    img_plot = np.rot90(np.fliplr(img_plot_raw.T))
    
    # correct exposure for each band individually
    img_plot_enhance = np.array(img_plot, copy=True)
    
    
    for band in range(3):
        
        # check max amount of a single value
        max_count_single_value = np.max(np.unique(img_plot, return_counts=True)[1])
        
        # if there are more than specific values set them as nan
        if max_count_single_value > max_single_value_count:
            no_data_value = img_plot.flatten()[np.argmax(np.unique(img_plot, return_counts=True)[1])]
            img_plot[img_plot == no_data_value] = np.nan 
            
        p_1, p_2 = np.nanpercentile(img_plot[:,:,band], (lower_percentile, upper_percentile))
        img_plot_enhance[:,:,band] = exposure.rescale_intensity(img_plot[:,:,band], 
                                                            in_range=(p_1, p_2), 
                                                            out_range = 'uint8')  

    return img_plot_enhance.astype('uint8')


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

    # creating model data dictionary
    model_band_info_dict = {}
    category_index = label_map_util.create_category_index_from_labelmap(args['label_file'], use_display_name=True)

    for model_file_name in os.listdir(args['model_dir']):
        model_band_info_dict[model_file_name] = list(map(int, input(f'Enter band no. seperated by comma for model: {model_file_name}\n').split(', ')))

    # Processing tif files
    for tif_file_name in os.listdir(args['input_dir']):
        tif_file_path = os.path.join(args['input_dir'], tif_file_name)
        
        op_img_np = tif_to_image_conversion(
                                        tif_file_path,
                                        [4,3,2]
                                        )

        # processing tif for each model
        for model_file_name in os.listdir(args['model_dir']):
            model_file_path = os.path.join(args['model_dir'], model_file_name)

            img_np = tif_to_image_conversion(
                                        tif_file_path,
                                        model_band_info_dict[model_file_name]
                                    )

            detection_graph = get_detection_graph(model_file_path)

            output_dict = run_inference_for_single_image(img_np, detection_graph)

            vis_util.visualize_boxes_and_labels_on_image_array(
                op_img_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=3,
                min_score_thresh=args['threshold'])
    
        im = Image.fromarray(op_img_np)
        im.save(os.path.join(args['output_dir'], tif_file_name.split('.')[0]+'.jpg'))
        print('Processed:', tif_file_name)
        

    



    