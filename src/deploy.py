import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import os
from dotenv import load_dotenv

# todo: you want logging in here?
# integrate deploy_model however

'''def deploy_model_to_sagemaker(bucket_name, model_file, role): # move to deploy.py?
    logger = setup_logging()
    logger.info("Creating SageMaker model and deploying endpoint")

    sagemaker_session = sagemaker.Session()
    model = TensorFlowModel(
        model_data=f's3://{bucket_name}/{model_file}',
        role=role,
        framework_version='2.3.0',
        sagemaker_session=sagemaker_session
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large'
    )
    logger.info("Deployment complete")
    return predictor'''

load_dotenv(dotenv_path='.env')

class Deploy:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucket_name = os.getenv('MODEL_BUCKET_NAME')
        self.role = os.getenv('AWS_IAM_ROLE')
        self.framework_version = '2.18.0'
        self.model_file = 'model.keras'
        self.bucket_url = self.s3_upload()
        self.url = self.deploy_model()

    def s3_upload(self):
        self.s3.upload_file(self.model_file, self.bucket_name, self.model_file)
        s3_url = f's3://{self.bucket_name}/{self.model_file}'
        return s3_url

    def deploy_model(self):
        sagemaker_session = sagemaker.Session()
        model = TensorFlowModel(
            model_data=self.model_s3_url,  # Use the stored S3 URL
            role=self.role,
            framework_version=self.framework_version,
            sagemaker_session=sagemaker_session
        )
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large'
        )
        # The endpoint name is used to construct the endpoint URL
        endpoint_url = f"https://runtime.sagemaker.{sagemaker_session.boto_region_name}.amazonaws.com/endpoints/{predictor.endpoint_name}/invocations"
        return endpoint_url

    def delete_endpoint(self, predictor):
        predictor.delete_endpoint()