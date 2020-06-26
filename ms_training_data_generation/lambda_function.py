"""
    Training data generation lambda
    Lambda get triggered when a config file is uploaded to s3
    This lambda will fetch filepath from event.
    Then Run the fargate task with updated env variables.
"""

import boto3

REGION = 'us-east-1'
FARGATE_CLUSTER = 'techtree'
FARGATE_TASK_DEF_NAME = 'MsTrainnig:1'
FARGATE_SUBNET_ID = 'subnet-69b85848'
CONTAINER_NAME = 'modeltrainning'

S3_LOG_FILE_UPLOAD_PATH = 'treetech-workflow/Logs/Ms-Trainning-Data-Generation'

def lambda_handler(event, context):

    # get event data
    event_record = event['Records']
    bucket_data = event_record[0]
    bucket_name = bucket_data['s3']['bucket']['name']
    file_name = bucket_data['s3']['object']['key']

    src_file_path = bucket_name + "/" + file_name

    # logs
    print('Event', event_record)
    print('bucket_data=', bucket_data)
    print('src file path =', src_file_path)

    # set env and run task using fargate service
    client = boto3.client('ecs', region_name=REGION)
    resp = client.run_task(
        cluster=FARGATE_CLUSTER,
        launchType='FARGATE',
        taskDefinition=FARGATE_TASK_DEF_NAME,
        count=1,
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
                            'name': 'S3_CONFIG_FILE_PATH',
                            'value': src_file_path
                        },
                        {
                            'name': 'S3_LOG_FILE_UPLOAD_PATH',
                            'value': S3_LOG_FILE_UPLOAD_PATH
                        }
                    ],
                },
            ],
        },
    )
    print('Fargate run task response:', str(resp))
    response = {
        "statusCode": 200,
        "body": str(resp)
    }
    return response
