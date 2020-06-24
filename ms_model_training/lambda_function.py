import boto3

REGION = 'us-east-1'
IMAGE_URI = '463836536571.dkr.ecr.us-east-1.amazonaws.com/techtree:Model-Training-DLC'
ROLE_ARN = 'arn:aws:iam::463836536571:role/service-role/AmazonSageMaker-ExecutionRole-20200622T204637'
OUTPUT_BUCKET_PATH = 's3://gcw-treetect-tree-detection-dev/temp_data'

def lambda_handler(event, context):

    #  get event data
    event_record = event['Records']
    bucket_data = event_record[0]
    file_name = bucket_data['s3']['object']['key']

    job_name = file_name.split('/')[-1].split('.')[0]

    # logs
    print('Event', event_record)
    print('bucket_data=', bucket_data)
    print('Job name', job_name)

    client = boto3.client('sagemaker', region_name=REGION)
    resp = client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            'TrainingImage': IMAGE_URI,
            'TrainingInputMode': 'File',
            'EnableSageMakerMetricsTimeSeries': True
        },
        RoleArn=ROLE_ARN,
        OutputDataConfig={
            'S3OutputPath': OUTPUT_BUCKET_PATH
        },
        ResourceConfig={
            'InstanceType': 'ml.g4dn.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 20,
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 86400,
        },
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=False,
        EnableManagedSpotTraining=False
        )

    print('Training job response:', str(resp))

    response = {
        "statusCode": 200,
        "body": str(resp)
    }
    return response