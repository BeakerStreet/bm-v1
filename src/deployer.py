import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import os
from dotenv import load_dotenv
import shutil
import tensorflow as tf
load_dotenv(dotenv_path='.env')

class Deploy:
    def __init__(self, model_path):
        self.s3 = boto3.client('s3')
        self.bucket_name = os.getenv('MODEL_BUCKET_NAME')
        self.role = os.getenv('AWS_IAM_ROLE')
        self.framework_version = '2.16'
        self.instance_type = 'ml.m5.large'
        self.instance_count = 1
        self.region = os.getenv('SAGEMAKER_REGION')
        self.model_file = model_path
        self.s3_url = None
        self.endpoint_name = 'bm-v1'
        
    def s3_upload(self):
        '''
        Converts model to SavedModel format 
        and uploads to S3
        '''

        # Convert the model to SavedModel format
        saved_model_dir = 'models/saved_model'
        model_archive = 'models/model.tar.gz'

        # Create directory as needed
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)

        # Save keras model as SavedModel
        model = tf.keras.models.load_model(self.model_file)
        model.save(saved_model_dir)

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
        predictor = model.deploy(
            initial_instance_count=self.instance_count,
            instance_type=self.instance_type,
            endpoint_name=self.endpoint_name
        )
        
        endpoint_url = f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{predictor.endpoint_name}/invocations"
        
        return endpoint_url
