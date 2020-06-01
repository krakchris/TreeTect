'''
    Script to automate training process
'''

# importing
import datetime
import json
import logging
import os
import subprocess
from subprocess import PIPE
import sys
import time

import shutil

#CONSTANTS
TRAIN_TEST_SPLIT_SCRIPT_PATH = os.path.join('..', '..', 'utils', 'data_processing', 'generate_train_test_split_from_csv.py')
TEST_PORTION = 0.2
CONVERT_TIF_INTO_JPG_CONVERSION_SCRIPT_PATH = os.path.join('..', '..', 'utils', 'data_processing', 'convert_tiff_into_jpeg.py')
TRAINING_CONFIG_JSON_PATH = os.path.join('..', 'training_config.json')
TEMP_DIR_PATH = os.path.join('..', 'temp_files')
LOG_FILE_PATH = os.path.join('..', 'training.log')
DATASET_DIR_PATH = os.path.join('..', 'dataset')
TIF_DIR_PATH = os.path.join(DATASET_DIR_PATH, 'tif_files')
IMAGE_DIR_PATH = os.path.join(DATASET_DIR_PATH, 'image_files')
TRAIN_TFRECORD_PATH = os.path.join(DATASET_DIR_PATH, 'train.record')
TEST_TFRECORD_PATH = os.path.join(DATASET_DIR_PATH, 'test.record')

#logger
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

logging.basicConfig(filename=LOG_FILE_PATH,
                filemode='a',
                format='%(asctime)s - %(message)s',
                level=logging.INFO,
                datefmt='%d-%b-%y %H:%M:%S')

def run_subprocess(command_list, input=None):
    '''
        Method to run command line process
        params:
            command_list: list of subprocesses command
            input: command line input
    '''
    if input:
        process_output = subprocess.run(command_list, input=input, stdout=sys.stdout, stderr=PIPE, encoding=ascii)
    else:
        process_output = subprocess.run(command_list, stdout=sys.stdout, stderr=PIPE)
    
    if 'Error' in process_output.stderr.decode('utf-8'):
        raise Exception(process_output.stderr.decode('utf-8'))

def s3_data_transfer(src, dest, is_dir):
    '''
        Method to download data from s3
        params:
            src : src path of a data
            dst : dest path of data
            is_dir : Bool value to define src data is file or directory
    '''

    if is_dir:
        run_subprocess(['aws', 's3', 'cp', src, dest, '--recursive'])

    else:
        run_subprocess(['aws', 's3', 'cp', src, dest])

def generate_training_data(s3_dataset_path, band):
    '''
        method to generate training_data for a specific dataset
        it download tif files dir from s3 and then convert tif into jpg for defined version
        and then save the train.record and test.record in the dataset folder
        params:
            dataset_path : path of the dataset
            band : band list in which the tif will get converted
    '''
    if os.path.exists(DATASET_DIR_PATH):
        shutil.rmtree(DATASET_DIR_PATH)

    os.makedirs(TIF_DIR_PATH)
    os.makedirs(IMAGE_DIR_PATH)

    #------------------------------Download tif files----------------------------------------------------

    print('-- Downloading tif data from', s3_dataset_path)
    logging.info('Downloading tif data from : {s3_dataset_path}')
    s3_data_transfer('s3://' + s3_dataset_path, TIF_DIR_PATH, True)

    # ----------------------------- Generate train test csv if not in Dir--------------------------            PENDING

    train_csv_path = os.path.join(TIF_DIR_PATH, 'train_labels.csv')
    test_csv_path = os.path.join(TIF_DIR_PATH, 'test_labels.csv')

    if not(os.path.exists(train_csv_path) or os.path.exists(test_csv_path)):
        logging.error('train/test csv does not exist')
        raise FileNotFoundError('train/test.csv')
        
        '''
        annotation_file_path = os.path.join(TIF_DIR_PATH, 'annotations.txt')

        if not os.path.exists(annotation_file_path):
            logging.error("FileNotFound: annotations.txt")
            raise FileNotFoundError('annotations.txt')

        else:
            logging.info('Generating train/test csv from annotation.txt')
            run_subprocess([
                        'python',
                        TRAIN_TEST_SPLIT_SCRIPT_PATH,
                        f'--csv_file={annotation_file_path}',
                        f'--output_dir={TIF_DIR_PATH}',
                        f'--test_portion={TEST_PORTION}'])'''

    # --------------------Convert tif file into jpg for band given in config and save into dir---------------

    print(f"-- Converting tif info jpg using band, {band}")
    logging.info(f"Converting tif info jpg using band, {band}")
    run_subprocess([
                'python',
                CONVERT_TIF_INTO_JPG_CONVERSION_SCRIPT_PATH,
                f'--input_dir={TIF_DIR_PATH}',
                f'--output_dir={IMAGE_DIR_PATH}']
                input=', '.join(band))

    # ---------------------Generating tf record--------------------------------------------------------------

    logging.info('Generating train tf records')
    print('-- Generating train tf records...')
    run_subprocess(['python',
                    'generate_tfrecord.py',
                    f'--csv_file={train_csv_path}',
                    f'--image_dir={os.path.join(IMAGE_DIR_PATH, '_'.join(band))}',
                    f'--output_path={TRAIN_TFRECORD_PATH}'])
        
    logging.info('Generating test tf records')
    print('-- Generating test tf records...')
    run_subprocess(['python',
                    'generate_tfrecord.py',
                    f'--csv_file={test_csv_path}',
                    f'--image_dir={os.path.join(IMAGE_DIR_PATH, '_'.join(band))}',
                    f'--output_path={TEST_TFRECORD_PATH}'])

if __name__ == "__main__":

    if os.path.exists(TEMP_DIR_PATH):
        logging.info('Removing Temp directory.')
        shutil.rmtree(TEMP_DIR_PATH)

    logging.info('Creating temp directory.')
    os.makedirs(TEMP_DIR_PATH)
    
    # --------------------------------loading json file-----------------------------------------

    logging.info('Reading config file.')
    with open(TRAINING_CONFIG_JSON_PATH, "r") as config_file:
        meta_data_json = json.load(config_file)

    # --------------------------------- iteration over dataset and start training ----------------------------
    for iteration_no, dataset_info in enumerate(meta_data_json['dataset']):
        
        logging.info(f'Generate training data for {iteration_no}th iteration ')
        generate_training_data(dataset_info['dataset_path'], meta_data_json['band'])
