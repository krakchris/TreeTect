"""
    script to download file from s3 and then start process and then upload it on s3
"""

# importing
import datetime
import glob
import json
import logging
import os
import subprocess
import sys

import shutil
import shortuuid

S3_CONFIG_FILE_PATH = os.environ['S3_CONFIG_FILE_PATH']
S3_LOG_FILE_UPLOAD_PATH = os.environ['S3_LOG_FILE_UPLOAD_PATH']
S3_COMPARISON_DATA_UPLOAD_PATH = os.environ['S3_COMPARISON_DATA_UPLOAD_PATH']

LOG_FILE_PATH = '../point_comparison.log'
ACTUAL_SHAPE_DIR = '../actual_shp_dir'
PREDICTED_SHAPE_DIR = '../predicted_shp_dir'
OUTPUT_DIR = '../op_dir'
POINT_DATA_COMPARISON_SCRIPT_PATH = 'point_data_comparison.py'
CONFIG_FILE_PATH = os.path.join(OUTPUT_DIR, 'point_comparison_config.json')

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
    logging.info(f"\n\n{'*'*100}\n\nRunning command :{' '.join(command_list)}\n{input}\n\n{'*'*100}\n\n")

    if input:
        process_output = subprocess.run(
            command_list,
            input=input,
            stdout=sys.stdout,
            stderr=sys.stdout,
            check=True)
    else:
        process_output = subprocess.run(
            command_list,
            stdout=sys.stdout,
            stderr=sys.stdout,
            check=True)

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

if __name__ == "__main__":
    try:
        status = 'failure'

        if os.path.exists(ACTUAL_SHAPE_DIR):
            shutil.rmtree(ACTUAL_SHAPE_DIR)

        if os.path.exists(PREDICTED_SHAPE_DIR):
            shutil.rmtree(PREDICTED_SHAPE_DIR)

        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

        os.makedirs(ACTUAL_SHAPE_DIR)
        os.makedirs(PREDICTED_SHAPE_DIR)
        os.makedirs(OUTPUT_DIR)

        # ------------------downloading config file from s3 to local----------
        logging.info('Downloading config file')
        print('Downloading Config file...')

        s3_data_transfer(
            's3://' + S3_CONFIG_FILE_PATH,
            CONFIG_FILE_PATH,
            False)

        # ------------------extracting meta information from config file------------
        logging.info('Extracting meta information')
        print('Extracting meta information...')

        with open(CONFIG_FILE_PATH, 'r') as config_file_obj:
            config = json.load(config_file_obj)

        # ----------------- download actual shape files s3 to local folder----------------------
        logging.info('Downloading actual shape files from s3')
        print('Downloading actual shape files from s3...')

        s3_data_transfer(
            's3://' + config['actual_points_dir_path'],
            ACTUAL_SHAPE_DIR,
            True)

        # ----------------- download predicted shape files s3 to local folder---------------------
        logging.info('Downloading predicted shape files from s3')
        print('Downloading predicted shape files from s3...')

        s3_data_transfer(
            's3://' + config['predicted_points_dir_path'],
            PREDICTED_SHAPE_DIR,
            True)

        # ------------------- run comparison process ------------------------------------
        logging.info('Run comparison of points')
        print('Run comparison of points...')

        actual_shp_file_path = glob.glob(os.path.join(ACTUAL_SHAPE_DIR, '*shp'))[0]
        predicted_shp_file_path = glob.glob(os.path.join(PREDICTED_SHAPE_DIR, '*shp'))[0]

        run_subprocess([
            'python3',
            POINT_DATA_COMPARISON_SCRIPT_PATH,
            f'--actual_shp_file={actual_shp_file_path}',
            f'--predicted_shp_file={predicted_shp_file_path}',
            f'--threshold={config["threshold"]}',
            f'--output_dir={OUTPUT_DIR}'])

        # -----------------upload files to s3-------------------------------------------
        logging.info('Uploading results to s3')
        print('Uploading result to s3...')

        s3_data_transfer(
            OUTPUT_DIR,
            's3://' + S3_COMPARISON_DATA_UPLOAD_PATH + '/' + str(datetime.datetime.now()),
            True)

        status = 'success'

    except Exception as e:
        status = 'failure'
        print('Failure:', str(e))

        logging.error(f"\n\n{'#'*100}\n\n{str(e)}\n\n{'#'*100}\n\n")

    finally:
        logging.info('Uploading log file to s3')
        print('Uploading log file to s3')

        s3_data_transfer(
            LOG_FILE_PATH,
            f"s3://{S3_LOG_FILE_UPLOAD_PATH}/{status}_point_comparison_{shortuuid.uuid()}.log",
            False)
