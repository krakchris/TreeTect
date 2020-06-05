"""
    script to download file from s3 and then start process and then upload it on s3
"""

# importing
import logging
import os
import subprocess
from subprocess import PIPE
import sys

import shortuuid
import shutil

S3_BIG_TIF_FILE_PATH = os.environ['S3_BIG_TIF_FILE_PATH']
S3_CHUNKED_TIF_DIR_PATH = os.environ['S3_CHUNKED_TIF_DIR_PATH']
S3_LOG_FILE_UPLOAD_PATH = os.environ['S3_LOG_FILE_UPLOAD_PATH']

LOG_FILE_PATH = '../training.log'
INPUT_DIR = '../inp_data'
OUTPUT_DIR = '../op_dir'

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
    logging.info(f"\n\n{'*'*100}'\n\n'Running command :{' '.join(command_list)}\n{input}\n\n{'*'*100}\n\n")

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
        status = 'success'

        if os.path.exists(INPUT_DIR):
            shutil.rmtree(INPUT_DIR)

        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

        os.makedirs(INPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        # ----------------- download file from s3 to local folder-----------------------------
        logging.info('Downloading big tif file from s3')
        print('Downloading big tif file from s3...')

        s3_data_transfer(
            's3://' + S3_BIG_TIF_FILE_PATH,
            INPUT_DIR,
            False)

        # ------------------- run chunking process ------------------------------------
        logging.info('Run chunking of file')
        print('Run chunking of file...')

        run_subprocess([
            'python',
            'file_chunker.py',
            f'--tif_dir={INPUT_DIR}',
            f'--output_dir={OUTPUT_DIR}'])

        # -----------------upload files to s3-------------------------------------------
        logging.info('Uploading chunked files to s3')
        print('Uploading chunked files to s3')

        s3_data_transfer(
            OUTPUT_DIR,
            's3://' + S3_CHUNKED_TIF_DIR_PATH,
            True)

    except Exception as e:
        status = 'failure'
        print('Failure:', str(e))

        logging.error(f"\n\n{'#'*100}\n\n{str(e)}\n\n{'#'*100}\n\n")

    finally:
        logging.info('Uploading log file to s3')
        print('Uploading log file to s3')

        s3_data_transfer(
            LOG_FILE_PATH,
            f"s3://{S3_LOG_FILE_UPLOAD_PATH}/{S3_BIG_TIF_FILE_PATH.split('/')[-1]}_{status}_chunking_{shortuuid.uuid()}.log",
            False)
