"""
    script to upload training_config.json to a fixed location on s3 with uuid
    command to run:
                python upload.py
"""

import sys
import subprocess
import uuid

CONFIG_DIR_S3_PATH = "gcw-treetect-tree-detection-dev/temp_data/config_files/training_config"
CONFIG_FILE_LOCAL_PATH = "training_config.json"

if __name__ == "__main__":

    dst_file_name = str(uuid.uuid4()) + '_' + CONFIG_FILE_LOCAL_PATH.split('/')[-1]
    dst_path = 's3://' + CONFIG_DIR_S3_PATH + '/' + dst_file_name

    subprocess.run(
        ['aws', 's3', 'cp', CONFIG_FILE_LOCAL_PATH, dst_path],
        stdout=sys.stdout,
        stderr=sys.stdout,
        check=True)

    print('File uploaded to:', dst_path)
