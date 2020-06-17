"""
    script to download required files from s3 and generate training data and upload back to s3
"""

# importing
import logging
import json
import os
import subprocess
import sys
import shutil

import shortuuid

S3_LOG_FILE_UPLOAD_PATH = os.environ['S3_LOG_FILE_UPLOAD_PATH']
S3_CONFIG_FILE_PATH = os.environ['S3_CONFIG_FILE_PATH']

TRAINING_DATA_GENERATION_SCRIPT_PATH = '../../utils/generate_training_data.py'
CSV_GENERATE_SCRIPT_PATH = '../../utils/generate_train_test_split_from_csv.py'

CONFIG_LOCAL_PATH = '../training_data_generation_config.json'
LOG_FILE_PATH = '../training.log'
TIF_DIR = '../tif_dir'
SHP_DIR = '../shp_dir'
OUTPUT_DIR = '../output_dir'

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

        if os.path.exists(TIF_DIR):
            shutil.rmtree(TIF_DIR)

        if os.path.exists(SHP_DIR):
            shutil.rmtree(SHP_DIR)

        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

        os.makedirs(TIF_DIR)
        os.makedirs(SHP_DIR)
        os.makedirs(OUTPUT_DIR)

        # ----------------download config file from s3 to local---------------------------
        logging.info('Downloading config file from s3')
        print('Downloading config file from s3...')

        s3_data_transfer(
            's3://' + S3_CONFIG_FILE_PATH,
            CONFIG_LOCAL_PATH,
            False)

        # -----------------------read config file ----------------------------------------
        with open(CONFIG_LOCAL_PATH, "r") as meta_file:
            meta_data_json = json.load(meta_file)

        # ----------------- download tif files from s3 to local folder-----------------------------
        logging.info('Downloading tif file from s3')
        print('Downloading tif file from s3...')

        s3_data_transfer(
            's3://' + meta_data_json['s3_tif_files_dir_path'],
            TIF_DIR,
            True)

        # ----------------- download shape file from s3 to local folder-----------------------------
        logging.info('Downloading shape files from s3')
        print('Downloading shape files from s3...')

        s3_data_transfer(
            's3://' + meta_data_json['s3_shape_files_dir_path'],
            SHP_DIR,
            True)

        # ---------------- run training data generation process------------------------------------
        logging.info('Running training data generation script')
        print('Running training data generation script...')

        run_subprocess(['python3',
                        TRAINING_DATA_GENERATION_SCRIPT_PATH,
                        f'--tif_dir={TIF_DIR}',
                        f'--shape_dir={SHP_DIR}',
                        f'--output_dir={OUTPUT_DIR}'])

        #  --------------------generate train test csv in the same dir -----------------------------
        logging.info('Running train test split script')
        print('Running train test split script...')

        run_subprocess(['python3',
                        CSV_GENERATE_SCRIPT_PATH,
                        f'--csv_file={os.path.join(OUTPUT_DIR, "annotations.txt")}',
                        f'--output_dir={OUTPUT_DIR}',
                        f'--test_portion={meta_data_json["csv_test_portion"]}'])

        #  ---------------------------upload all data on s3--------------------------------------
        logging.info('Uploading files to s3')
        print('Uploading files to s3...')

        s3_data_transfer(
            OUTPUT_DIR,
            's3://' + meta_data_json['s3_training_data_upload_dir_path'] + '/' + meta_data_json['s3_tif_files_dir_path'].split('/')[-1],
            True)

        status = 'success'

    except Exception as e:

        print('Failure:', str(e))

        logging.error(f"\n\n{'#'*100}\n\n{str(e)}\n\n{'#'*100}\n\n")

    finally:
        logging.info('Uploading log file to s3')
        print('Uploading log file to s3')

        s3_data_transfer(
            LOG_FILE_PATH,
            f"s3://{S3_LOG_FILE_UPLOAD_PATH}/{status}_'training_data_generation'_{shortuuid.uuid()}.log",
            False)
