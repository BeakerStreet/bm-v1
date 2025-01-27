import logging
import boto3
import numpy as np
from src.dataset import Dataset
from tensorflow.keras.models import load_model
import sys
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from src.deploy import Deploy
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='.env')

s3 = boto3.client('s3')

def setup_logging(): # do we want this? Or should it be in the deploy.py file?
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def upload_model_to_s3(model_file, bucket_name): # move to deploy.py?
    logger = setup_logging()
    logger.info(f"Uploading {model_file} to S3 bucket {bucket_name}")
    s3.upload_file(model_file, bucket_name, model_file)
    logger.info("Upload complete")

def deploy_model_to_sagemaker(bucket_name, model_file, role): # move to deploy.py?
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
    return predictor

def train():
    logger = setup_logging()
    logger.info("=== Training ===")

    # Load the dataset
    dataset = Dataset(dataset_path='data/dataset.json', image_folder='data/assets')

    # The model is already built and compiled in the Dataset class
    model = dataset.model

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([dataset.image_embeddings, dataset.text_embeddings], dataset.labels, epochs=10, batch_size=32)

    # Save the model
    model.save('model.keras')

    logger.info("=== Training Complete ===")

def predict():
    logger = setup_logging()
    logger.info("=== Predicting ===")

    MODEL_BUCKET_NAME = os.getenv('MODEL_BUCKET_NAME')
    AWS_IAM_ROLE = os.getenv('AWS_IAM_ROLE')

    # Load the dataset for prediction
    dataset = Dataset(dataset_path='data/dataset.json', image_folder='data/assets')

    # Initialize Deploy class
    deployer = Deploy(MODEL_BUCKET_NAME, AWS_IAM_ROLE)

    # Upload model to S3
    logger.info(f"Uploading {deployer.model_file} to S3 bucket {deployer.bucket_name}")
    deployer.s3.upload_file(deployer.model_file, deployer.bucket_name, deployer.model_file)
    logger.info("Upload complete")

    # Deploy model to SageMaker
    logger.info("Creating SageMaker model and deploying endpoint")
    sagemaker_session = sagemaker.Session()
    model = TensorFlowModel(
        model_data=f's3://{deployer.bucket_name}/{deployer.model_file}',
        role=deployer.role,
        framework_version=deployer.framework_version,
        sagemaker_session=sagemaker_session
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large'
    )
    logger.info("Deployment complete")

    # Make a prediction
    logger.info("Invoking SageMaker endpoint for prediction")
    prediction = predictor.predict([image_data, text_data])
    logger.info(f"Prediction: {prediction}")
    
    print(prediction)

    # Clean up
    logger.info("Deleting SageMaker endpoint")
    predictor.delete_endpoint()
    logger.info("Endpoint deleted")

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <train|predict>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        train()
    elif command == "predict":
        predict()
    else:
        print("Invalid command. Use 'train' or 'predict'.")
        sys.exit(1)

if __name__ == "__main__":
    main()