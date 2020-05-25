'''
    Script to automate ensembling process
'''

# importing
import datetime
import json
import os
import subprocess
from subprocess import PIPE
import sys
import time

import shutil

#CONSTANTS
META_DATA_JSON_PATH = '../ensemble_meta_data.json'
TEMP_DIR_PATH = '../temp_files'
TIF_DIR_PATH = os.path.join(TEMP_DIR_PATH, 'tif_dir')
MODEL_DIR_PATH = os.path.join(TEMP_DIR_PATH, 'model_files')
TEMP_DOWNLOAD_PATH = os.path.join(TEMP_DIR_PATH, 'temp_download/')
S3_MODEL_DIR_BASE_PATH = 'gcw-treetect-tree-detection-dev/Models/worldview/development'
ENSEMBLE_OUTPUT_DIR_PATH = os.path.join(TEMP_DIR_PATH, 'ensemble_output')
POINT_DATA_DIR_PATH = os.path.join(ENSEMBLE_OUTPUT_DIR_PATH, 'point_data')
COMBINED_BOX_SHAPE_FILE_DIR = os.path.join(ENSEMBLE_OUTPUT_DIR_PATH, 'combined_box_shape_file')
COMBINED_POINT_SHAPE_FILE_DIR = os.path.join(ENSEMBLE_OUTPUT_DIR_PATH, 'combined_point_shape_file')
LABEL_FILE_PATH = '../../dataset/label_map.pbtxt'
THRESHOLD = 0.5

def run_subprocess(command_list):
    '''
        Method to run command line process
        params:
            command_list: list of subprocesses command
    '''
    process_output = subprocess.run(command_list, stdout=sys.stdout, stderr=PIPE)
    if process_output.stderr:
        raise Exception(process_output.stderr)

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

def get_meta_dict(meta_file_path):
    '''
        method to read data from text file and convert it into dictionary
        params:
            meta_file_path: path to the text meta data file
        return meta data dictionary
    '''
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f'{meta_file_path}')

    meta_data_dict = {}

    with open(meta_file_path, 'r') as meta_file:

        for line in meta_file:
            key, value = (line.strip()).split(':')
            meta_data_dict[key] = value

    return meta_data_dict

if __name__ == "__main__":

    if os.path.exists(TEMP_DIR_PATH):
        shutil.rmtree(TEMP_DIR_PATH)

    os.makedirs(TEMP_DIR_PATH)
    os.makedirs(TIF_DIR_PATH)
    os.makedirs(MODEL_DIR_PATH)
    os.makedirs(ENSEMBLE_OUTPUT_DIR_PATH)
    os.makedirs(POINT_DATA_DIR_PATH)
    os.makedirs(COMBINED_BOX_SHAPE_FILE_DIR)
    os.makedirs(COMBINED_POINT_SHAPE_FILE_DIR)

    # --------------------------------loading jaon file-----------------------------------------

    with open(META_DATA_JSON_PATH, "r") as meta_file:
        meta_data_json = json.load(meta_file)

    # -------------------------------download tif from s3 --------------------------------------
    print('downloading tif files...')
    s3_data_transfer('s3://' + meta_data_json['tif_dir_path'], TIF_DIR_PATH, True)

    # ------------------------------ download model file from s3 ------------------------------

    print('downloading model_files...')
    for model_version in meta_data_json['model_versions']:

        shutil.rmtree(TEMP_DOWNLOAD_PATH)
        os.makedirs(TEMP_DOWNLOAD_PATH)

        # download meta file
        model_meta_file_path = S3_MODEL_DIR_BASE_PATH + f'/{model_version}/meta_data.txt'
        s3_data_transfer('s3://' + model_meta_file_path, TEMP_DOWNLOAD_PATH, False)

        # download modelfrozen graph
        model_frozen_graph_path = S3_MODEL_DIR_BASE_PATH + f'/{model_version}/output_inference_graph/frozen_inference_graph.pb'
        s3_data_transfer('s3://' + model_frozen_graph_path, TEMP_DOWNLOAD_PATH, False)

        # move the frozen graph to the model files with new name
        meta_data_dict = get_meta_dict(os.path.join(TEMP_DOWNLOAD_PATH, 'meta_data.txt'))
        dst_path = os.path.join(MODEL_DIR_PATH,
                                '_'.join(meta_data_dict['band'].split(', ')) + '_frozen_inference_graph.pb')
        shutil.move(os.path.join(TEMP_DOWNLOAD_PATH, 'frozen_inference_graph.pb'), dst_path)

    # ----------------------------running ensemble script --------------------------------------------

    print('running ensembling process...')
    run_subprocess(['python',
                    '../utils/models_processing/ensemble.py',
                    f'--model_dir={MODEL_DIR_PATH}',
                    f'--input_dir={TIF_DIR_PATH}',
                    f'--output_dir={ENSEMBLE_OUTPUT_DIR_PATH}',
                    f'--label_file={LABEL_FILE_PATH}',
                    f'--threshold={THRESHOLD}'])

    # --------------------------- generating point data ...............................................

    print('generating point data...')
    run_subprocess(['python',
                    'generate_point_data.py',
                    f'--input_dir={TIF_DIR_PATH}',
                    f'--csv_file={os.path.join(ENSEMBLE_OUTPUT_DIR_PATH, "annotations.csv")}',
                    f'--output_dir={POINT_DATA_DIR_PATH}'])

    # --------------------------- combined shape file -------------------------------------------------

    print('combining shape files...')
    run_subprocess(['python',
                    'combined_shape_files.py',
                    f'--input_dir={os.path.join(ENSEMBLE_OUTPUT_DIR_PATH, "inference_shape_files")}',
                    f'--output_dir={COMBINED_BOX_SHAPE_FILE_DIR}'])

    run_subprocess(['python',
                    'combined_shape_files.py',
                    f'--input_dir={POINT_DATA_DIR_PATH}',
                    f'--output_dir={COMBINED_POINT_SHAPE_FILE_DIR}'])

    # ---------------------------- combine and upload all data-------------------------------------------

    print('uploading data on s3...')
    shutil.copy(META_DATA_JSON_PATH, ENSEMBLE_OUTPUT_DIR_PATH)

    NEW_ENSEMBLED_DATA_PATH = os.path.join(TEMP_DIR_PATH, str(datetime.datetime.now()))
    os.rename(ENSEMBLE_OUTPUT_DIR_PATH, NEW_ENSEMBLED_DATA_PATH)

    s3_data_transfer(NEW_ENSEMBLED_DATA_PATH, 's3://' + meta_data_json['s3_data_upload_path'], True)

    for i in range(5):
        print(f'system will shut down in {5-i} min')
        time.sleep(60)

    shutil.rmtree(TEMP_DIR_PATH)

    run_subprocess(['sudo', 'poweroff'])
