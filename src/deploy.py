import logging
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

class Deploy:
    def __init__(self, bucket_name, role, framework_version='2.18.0'):
        self.s3 = boto3.client('s3')
        self.bucket_name = os.getenv('MODEL_BUCKET_NAME')
        self.role = os.getenv('AWS_IAM_ROLE')
        self.framework_version = framework_version
        self.model_file = 'model.keras'

    def upload_to_s3(self):
        self.logger.info(f"Uploading {self.model_file} to S3 bucket {self.bucket_name}")
        self.s3.upload_file(self.model_file, self.bucket_name, self.model_file)
        self.logger.info("Upload complete")

    def deploy_model(self):
        self.logger.info("Creating SageMaker model and deploying endpoint")
        sagemaker_session = sagemaker.Session()
        model = TensorFlowModel(
            model_data=f's3://{self.bucket_name}/{self.model_file}',
            role=self.role,
            framework_version=self.framework_version,
            sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large'
        )
        self.logger.info("Deployment complete")
        return predictor

    def delete_endpoint(self, predictor):
        self.logger.info("Deleting SageMaker endpoint")
        predictor.delete_endpoint()
        self.logger.info("Endpoint deleted") 