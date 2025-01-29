import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import os
from dotenv import load_dotenv
import shutil
import tensorflow as tf
from botocore.exceptions import ClientError
from datetime import datetime
load_dotenv(dotenv_path='.env')

class Deploy:
    def __init__(self, model_path):
        self.s3 = boto3.client('s3')
        self.sagemaker_client = boto3.client('sagemaker')
        self.bucket_name = os.getenv('MODEL_BUCKET_NAME')
        self.role = os.getenv('AWS_IAM_ROLE')
        self.framework_version = '2.16'
        self.instance_type = 'ml.m5.large'
        self.instance_count = 1
        self.region = os.getenv('SAGEMAKER_REGION')
        self.model_file = model_path
        self.s3_url = None
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.endpoint_name = f'bm-v1-{timestamp}'
        
    def s3_upload(self):
        '''
        Converts model to SavedModel format 
        and uploads to S3
        '''

        # Define the directory and archive paths
        saved_model_dir = 'models/saved_model'
        model_archive = 'models/model.tar.gz'

        # Create directory as needed
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)

        # Load the Keras model
        model = tf.keras.models.load_model(self.model_file)

        # Save the model in TensorFlow SavedModel format
        tf.saved_model.save(model, saved_model_dir)  # Use tf.saved_model.save

        # Create a tar.gz archive of the SavedModel directory
        shutil.make_archive('models/model', 'gztar', 'models', 'saved_model')

        # Upload the archive to S3
        self.s3.upload_file(model_archive, self.bucket_name, 'model/model.tar.gz')
        self.s3_url = f's3://{self.bucket_name}/model/model.tar.gz'
        
        return self.s3_url

    def deploy_model(self):
        '''
        Deploys the model to SageMaker
        '''

        sagemaker_session = sagemaker.Session()

        model = TensorFlowModel(
            model_data=self.s3_url,
            role=self.role,
            framework_version=self.framework_version,
            sagemaker_session=sagemaker_session
        )

        # Check if the endpoint configuration already exists
        try:
            self.sagemaker_client.describe_endpoint_config(EndpointConfigName=self.endpoint_name)
            print(f"Endpoint configuration '{self.endpoint_name}' already exists.")
            # Optionally, you can delete the existing configuration
            # self.sagemaker_client.delete_endpoint_config(EndpointConfigName=self.endpoint_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                print(f"Endpoint configuration '{self.endpoint_name}' does not exist. Creating a new one.")
            else:
                raise

        predictor = model.deploy(
            initial_instance_count=self.instance_count,
            instance_type=self.instance_type,
            endpoint_name=self.endpoint_name
        )
        
        endpoint_url = f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{predictor.endpoint_name}/invocations"
        
        return endpoint_url
