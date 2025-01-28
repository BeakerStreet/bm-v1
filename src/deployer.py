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
        self.framework_version = '2.16'
        self.instance_type = 'ml.m5.large'
        self.instance_count = 1
        self.region = os.getenv('SAGEMAKER_REGION')
        self.model_file = model_path
        
    def s3_upload(self):
        '''
        Converts model.keras file to tar.gz archive 
        and uploads it to AWS S3 bucket
        '''

        # Archive the model
        model_archive = 'models/model.tar.gz'
        os.system(f'tar -czvf {model_archive} {self.model_file}')
        
        # Upload
        self.s3.upload_file(model_archive, self.bucket_name, model_archive)
        s3_url = f's3://{self.bucket_name}/{model_archive}'
        
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
