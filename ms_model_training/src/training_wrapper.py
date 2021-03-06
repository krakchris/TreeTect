'''
    Script to automate training process
'''

# importing
import glob
import json
import logging
import os
import subprocess
import sys

import pandas as pd
import shortuuid
import shutil
import tensorflow as tf

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

#CONSTANTS
CURRENT_PATH = os.getcwd()

# MODEL_BASE_ARCHITECTURE_S3_PATH = os.environ['MODEL_BASE_ARCHITECTURE_S3_PATH']
# LABLE_FILE_S3_PATH = os.environ['LABLE_FILE_S3_PATH']
# S3_LOG_FILE_UPLOAD_PATH = os.environ['S3_LOG_FILE_UPLOAD_PATH']
# S3_MODEL_UPLOAD_PATH = os.environ['S3_MODEL_UPLOAD_PATH']
# CONFIG_FILE_S3_PATH = os.environ['CONFIG_FILE_S3_PATH']

CONFIG_FILE_S3_PATH = 'treetech-workflow/Config/model_training/' + os.environ['TRAINING_JOB_NAME'] + '.json'

DATASET_DIR_PATH = os.path.join(CURRENT_PATH, '..', 'dataset')
TIF_DIR_PATH = os.path.join(DATASET_DIR_PATH, 'tif_files')
IMAGE_DIR_PATH = os.path.join(DATASET_DIR_PATH, 'image_files')

CONVERT_TIF_INTO_JPG_CONVERSION_SCRIPT_PATH = '../../utils/convert_tiff_into_jpeg.py'

TRAINING_CONFIG_JSON_PATH = os.path.join('..', 'training_config.json')
LOG_FILE_PATH = os.path.join('..', 'training.log')
TRAIN_TFRECORD_PATH = os.path.join(DATASET_DIR_PATH, 'train.record')
TEST_TFRECORD_PATH = os.path.join(DATASET_DIR_PATH, 'test.record')
LABEL_MAP_PATH = os.path.join(DATASET_DIR_PATH, 'label_map.pbtxt')
TRAIN_CSV_PATH = os.path.join(TIF_DIR_PATH, 'train_labels.csv')
TEST_CSV_PATH = os.path.join(TIF_DIR_PATH, 'test_labels.csv')

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

    #------------------------------Download tif files-----------------------------------------------

    print('-- Downloading tif data from', s3_dataset_path)
    logging.info(f'Downloading tif data from : {s3_dataset_path}')
    s3_data_transfer('s3://' + s3_dataset_path, TIF_DIR_PATH, True)

    # ----------------------------- Check CSV file in dir-------------------------------------------

    if not(os.path.exists(TRAIN_CSV_PATH) or os.path.exists(TEST_CSV_PATH)):
        logging.error('train/test csv does not exist')
        raise FileNotFoundError('train/test.csv')

    # --------------------Convert tif file into jpg for band given in config and save into dir------

    print(f"-- Converting tif info jpg using band, {band}")
    logging.info(f"Converting tif info jpg using band, {band}")
    run_subprocess([
                'python3',
                CONVERT_TIF_INTO_JPG_CONVERSION_SCRIPT_PATH,
                f'--input_dir={TIF_DIR_PATH}',
                f'--output_dir={IMAGE_DIR_PATH}'],
                   input=(', '.join(band) + '\nz\n').encode())

    # ---------------------Generating tf record-----------------------------------------------------

    logging.info('Generating train tf records')
    print('-- Generating train tf records...')
    run_subprocess(['python3',
                    'generate_tfrecord.py',
                    f'--csv_file={TRAIN_CSV_PATH}',
                    f'--image_dir={os.path.join(IMAGE_DIR_PATH, "_".join(band))}',
                    f'--output_path={TRAIN_TFRECORD_PATH}'])

    logging.info('Generating test tf records')
    print('-- Generating test tf records...')
    run_subprocess(['python3',
                    'generate_tfrecord.py',
                    f'--csv_file={TEST_CSV_PATH}',
                    f'--image_dir={os.path.join(IMAGE_DIR_PATH, "_".join(band))}',
                    f'--output_path={TEST_TFRECORD_PATH}'])

    # -----------------------downloading label_map.pbtxt file --------------------------------------

    logging.info('Downloading label file from s3')
    print('-- Downloading label file...')
    s3_data_transfer('s3://' + LABLE_FILE_S3_PATH, LABEL_MAP_PATH, False)

if __name__ == "__main__":

    try:
        status = 'failure'
        # ----------------------------download training_config.json file from s3------------------
        logging.info(f'Downloading {CONFIG_FILE_S3_PATH}')
        print(f'Downloading {CONFIG_FILE_S3_PATH}...')

        s3_data_transfer('s3://' + CONFIG_FILE_S3_PATH, TRAINING_CONFIG_JSON_PATH, False)

        # --------------------------------loading json file-----------------------------------------

        logging.info('Reading config file.')
        with open(TRAINING_CONFIG_JSON_PATH, "r") as config_file:
            meta_data_json = json.load(config_file)

            # setting up constant variables
            MODEL_BASE_ARCHITECTURE_S3_PATH = meta_data_json['MODEL_BASE_ARCHITECTURE_S3_PATH']
            LABLE_FILE_S3_PATH = meta_data_json['LABLE_FILE_S3_PATH']
            S3_LOG_FILE_UPLOAD_PATH = meta_data_json['S3_LOG_FILE_UPLOAD_PATH']
            S3_MODEL_UPLOAD_PATH = meta_data_json['S3_MODEL_UPLOAD_PATH']

        # ----------------------download base model path and rename according to version------------

        logging.info('Downloading base model files...')
        print('Downloading base model files...')

        model_files_dir = os.path.join(
            CURRENT_PATH,
            '..',
            f"{meta_data_json['model_version']}_{meta_data_json['model_architecture']}")

        if meta_data_json['is_transferlearn']:
            s3_data_transfer(
                's3://' + os.path.join(S3_MODEL_UPLOAD_PATH,
                                       meta_data_json['transfer_learn_from']),
                model_files_dir,
                True)

            checkpoint_txt_log_path = os.path.join(model_files_dir, 'training', 'checkpoint')

            res = ''
            with open(checkpoint_txt_log_path, 'r') as file_obj:
                for line in file_obj:
                    key, value = (line.strip().split(': '))
                    value = os.path.join(model_files_dir, 'training', value.split('/')[-1][:-1])
                    res += key + ': ' + '"' + value + '"\n'

            with open(checkpoint_txt_log_path, 'w') as file_obj:
                file_obj.write(res)

            shutil.rmtree(os.path.join(model_files_dir, 'eval'))
            shutil.rmtree(os.path.join(model_files_dir, 'output_inference_graph'))

        else:
            s3_data_transfer(
                's3://' + os.path.join(MODEL_BASE_ARCHITECTURE_S3_PATH,
                                       meta_data_json['model_architecture']),
                model_files_dir,
                True)

        # --------------------------- iteration over dataset and start training --------------------
        for iteration_no, dataset_info in enumerate(meta_data_json['dataset']):

            logging.info(f'Generate training data for {iteration_no}th iteration ')
            generate_training_data(dataset_info['dataset_path'], meta_data_json['band'])

            # ----------------------------------update config---------------------------------------

            print(f'iteration: {iteration_no}  :Updating model config file...')
            logging.info(f'iteration: {iteration_no}  :Updating model config file')

            model_config_path = glob.glob(os.path.join(model_files_dir, '*.config'))[0]

            pipeline = pipeline_pb2.TrainEvalPipelineConfig()

            with tf.gfile.GFile(model_config_path, "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, pipeline)

            pipeline.train_config.fine_tune_checkpoint = os.path.join(
                model_files_dir,
                f"base_model_{meta_data_json['model_architecture']}",
                'model.ckpt')

            if meta_data_json['is_transferlearn']:
                pipeline.train_config.num_steps += dataset_info['training_steps']
            else:
                pipeline.train_config.num_steps = dataset_info['training_steps']

            pipeline.train_config.batch_size = meta_data_json['batch_size']

            pipeline.train_input_reader.label_map_path = LABEL_MAP_PATH
            pipeline.train_input_reader.tf_record_input_reader.input_path[:] = [TRAIN_TFRECORD_PATH]

            df = pd.read_csv(TEST_CSV_PATH)
            unique_file_count = df['filename'].unique().size

            vis_dir = os.path.join(model_files_dir, 'eval', 'visualizations')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)

            pipeline.eval_config.max_evals = 1
            pipeline.eval_config.num_visualizations = unique_file_count
            pipeline.eval_config.num_examples = unique_file_count
            pipeline.eval_config.visualization_export_dir = vis_dir

            pipeline.eval_input_reader[0].label_map_path = LABEL_MAP_PATH
            pipeline.eval_input_reader[0].tf_record_input_reader.input_path[:] = [TEST_TFRECORD_PATH]

            config_text = text_format.MessageToString(pipeline)
            with tf.gfile.Open(model_config_path, "wb") as f:
                f.write(config_text)

            # ------------------------------------------training------------------------------------

            print(f'iteration: {iteration_no}  :Start training..')
            logging.info(f'iteration: {iteration_no}  :Start training')

            run_subprocess([
                'python3',
                'train.py',
                '--logtostderr',
                f'--train_dir={os.path.join(model_files_dir, "training")}',
                f'--pipeline_config_path={model_config_path}'])

            # ------------------------------------copying training checkpoints----------------------

            print(f'iteration: {iteration_no}  :Copying checkpoint data...')
            logging.info(f'iteration: {iteration_no}  :Copying checkpoint data')

            chk_dir_path = os.path.join(model_files_dir,
                                         f"{iteration_no}_checkpoint_{meta_data_json['model_architecture']}")

            if os.path.exists(chk_dir_path):
                shutil.rmtree(chk_dir_path)

            shutil.copytree(os.path.join(model_files_dir, 'training'), chk_dir_path)


        shutil.rmtree(chk_dir_path) #delete latest checkpoint as it is same as in training dir

        # --------------------------------running evaluation for latest checkpoints-----------------

        print('Running evaluation...')
        logging.info('Running evaluation')
        run_subprocess([
            'python3',
            'modified_eval.py',
            '--logtostderr',
            f'--pipeline_config_path={model_config_path}',
            f'--checkpoint_dir={os.path.join(model_files_dir, "training")}',
            f'--eval_dir={os.path.join(model_files_dir, "eval")}',
            f'--output_json_path={os.path.join(model_files_dir, "evaluation_results.json")}'])

        # ----------------------------------freeze model-------------------------------------------

        print('Freezing model graph...')
        logging.info('Freezing model graph')
        pipeline_config_path = os.path.join(model_files_dir, 'training', 'pipeline.config')

        max_checkpoint_number = max([int(file_name.split('.')[1].split('-')[-1])
                                     for file_name in os.listdir(os.path.join(model_files_dir,
                                                                              'training'))
                                     if file_name.endswith(('.index'))])

        trained_checkpoint_prefix = os.path.join(model_files_dir,
                                                 'training',
                                                 f"model.ckpt-{max_checkpoint_number}")

        run_subprocess(['python3',
                        'export_inference_graph.py',
                        '--input_type=image_tensor',
                        f'--pipeline_config_path={pipeline_config_path}',
                        f'--trained_checkpoint_prefix={trained_checkpoint_prefix}',
                        f'--output_directory={os.path.join(model_files_dir, "output_inference_graph")}'])

        # ---------------------------------creating meta file---------------------------------------

        print('creating meta.txt file...')
        logging.info('creating meta.txt file.')

        with open(os.path.join(model_files_dir, 'meta_data.txt'), 'w') as meta_txt_file:

            meta_txt_file.write(f'model_version:{meta_data_json["model_version"]}\n')
            meta_txt_file.write(f'model_architecture:{meta_data_json["model_architecture"]}\n')
            meta_txt_file.write(f'band:{", ".join(meta_data_json["band"])}\n')

            for index, dataset_info in enumerate(meta_data_json['dataset']):
                meta_txt_file.write(f'dataset_path_{index}:{dataset_info["dataset_path"]}\n')
                meta_txt_file.write(f'training_steps_{index}:{dataset_info["training_steps"]}\n')

        # --------------------------------remove unnecessary files ---------------------------------

        print('Removing base folder...')
        logging.info('Removing base folder')
        shutil.rmtree(os.path.join(model_files_dir,
                                   f"base_model_{meta_data_json['model_architecture']}"))

        # ---------------------------------copy log file to the model folder------------------------

        print('Copying log file...')
        logging.info('Copying log file into model dir')
        shutil.copy(LOG_FILE_PATH, os.path.join(model_files_dir))

        # --------------------------------uploading data to s3--------------------------------------

        print('Uploading training files to s3...')
        logging.info('Uploading training files to s3')
        s3_data_transfer(model_files_dir,
                         's3://'+ S3_MODEL_UPLOAD_PATH + '/' + model_files_dir.split('/')[-1],
                         True)

        status = 'success'

    except Exception as e:

        print(str(e))
        logging.error(f"\n\n{'#'*100}'\n\n'{str(e)}\n\n{'#'*100}\n\n")

    finally:

        # ---------------------------------send notification and shutdown---------------------------

        print('uploading log file to s3....')
        logging.info('uploading log file to s3')
        s3_data_transfer(
            LOG_FILE_PATH,
            f"s3://{S3_LOG_FILE_UPLOAD_PATH}/{meta_data_json['model_version']}_{status}_taining_{shortuuid.uuid()}.log",
            False)
