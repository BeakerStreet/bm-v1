import logging
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

class Deploy:
    def __init__(self, bucket_name, role, framework_version='2.3.0'):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.role = role
        self.framework_version = framework_version
        self.logger = self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

    def upload_model_to_s3(self, model_file):
        self.logger.info(f"Uploading {model_file} to S3 bucket {self.bucket_name}")
        self.s3.upload_file(model_file, self.bucket_name, model_file)
        self.logger.info("Upload complete")

    def deploy_model(self, model_file):
        self.logger.info("Creating SageMaker model and deploying endpoint")
        sagemaker_session = sagemaker.Session()
        model = TensorFlowModel(
            model_data=f's3://{self.bucket_name}/{model_file}',
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