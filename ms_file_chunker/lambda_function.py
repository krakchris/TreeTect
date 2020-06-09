import json
import logging
import os
import time
import uuid
from datetime import datetime
import boto3

# config
REGION = 'us-east-1'
FARGATE_CLUSTER = 'techtree'
FARGATE_TASK_DEF_NAME = 'Filechunker:2'
FARGATE_SUBNET_ID = 'subnet-69b85848'
CONTAINER_NAME = 'filechunker'

def lambda_handler(event, context):
    
    client = boto3.client('ecs', region_name=REGION)
    resp = client.run_task(
        cluster=FARGATE_CLUSTER,
        launchType = 'FARGATE',
        taskDefinition=FARGATE_TASK_DEF_NAME,
        count = 1,
        platformVersion='LATEST',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [
                    FARGATE_SUBNET_ID,
                ],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': CONTAINER_NAME,
                    'environment': [
                        {
                            'name': 'S3_BIG_TIF_FILE_PATH',
                            'value': 'gcw-treetect-common-input-data-dev/input_data/source_data/worldview/Amsterdam/modified_merged.tif'
                        },
                        {
                            'name': 'S3_CHUNKED_TIF_DIR_PATH',
                            'value': 'gcw-treetect-tree-detection-dev/temp_data/chunked_tif_data'
                        },
                        {
                            'name': 'S3_LOG_FILE_UPLOAD_PATH',
                            'value': 'gcw-treetect-tree-detection-dev/training_logs'
                        },
                    ],
                },
            ],
        },
    )

    response = {
        "statusCode": 200,
        "body": str(resp)
    }
    return response
