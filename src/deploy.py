import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

class Deploy:
    def __init__(self, model_path):
        self.s3 = boto3.client('s3')
        self.bucket_name = os.getenv('MODEL_BUCKET_NAME')
        self.role = os.getenv('AWS_IAM_ROLE')
        self.framework_version = '2.18.0'
        self.instance_type = 'ml.m5.large'
        self.instance_count = 1
        self.region = os.getenv('SAGEMAKER_REGION')
        self.model_file = model_path
        self.bucket_url = self.s3_upload()
        self.url = self.deploy_model()
        
    def s3_upload(self):
        '''
        Uploads the model.keras file to the S3 bucket
        '''
        
        self.s3.upload_file(self.model_file, self.bucket_name, self.model_file)
        s3_url = f's3://{self.bucket_name}/{self.model_file}'
        
        return s3_url

    def deploy_model(self):
        '''
        Deploys the model to SageMaker
        '''

        sagemaker_session = sagemaker.Session()

        model = TensorFlowModel(
            model_data=self.bucket_url,
            role=self.role,
            framework_version=self.framework_version,
            sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(
            initial_instance_count=self.instance_count,
            instance_type=self.instance_type
        )
        
        endpoint_url = f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{predictor.endpoint_name}/invocations"
        
        return endpoint_url
