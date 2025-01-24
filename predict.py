import logging
import boto3
import numpy as np
from src.dataset import Dataset
from tensorflow.keras.models import load_model
import sys
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from src.deploy import Deploy

s3 = boto3.client('s3')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def upload_model_to_s3(model_file, bucket_name):
    logger = setup_logging()
    logger.info(f"Uploading {model_file} to S3 bucket {bucket_name}")
    s3.upload_file(model_file, bucket_name, model_file)
    logger.info("Upload complete")

def deploy_model_to_sagemaker(bucket_name, model_file, role):
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

def predict_with_sagemaker(predictor, image_data, text_data):
    logger = setup_logging()
    logger.info("Invoking SageMaker endpoint for prediction")
    prediction = predictor.predict([image_data, text_data])
    logger.info(f"Prediction: {prediction}")
    return prediction

def train():
    logger = setup_logging()
    logger.info("=== Starting Train ===")

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

    logger.info("=== Training Process Complete ===")

def predict():
    logger = setup_logging()
    logger.info("=== Starting Predict ===")

    # Load the dataset for prediction
    dataset = Dataset(dataset_path='data/dataset.json', image_folder='data/assets/predict')

    # Select a random image and its corresponding text embedding
    random_index = np.random.randint(len(dataset.image_embeddings))
    image_data = np.expand_dims(dataset.image_embeddings[random_index], axis=0)
    text_data = np.expand_dims(dataset.text_embeddings[random_index], axis=0)

    # Initialize Deploy class
    bucket_name = 'your-s3-bucket-name'
    role = 'your-sagemaker-execution-role'
    deployer = Deploy(bucket_name, role)

    # Upload model to S3
    model_file = 'model.keras'
    deployer.upload_model_to_s3(model_file)

    # Deploy model to SageMaker
    predictor = deployer.deploy_model(model_file)

    # Make a prediction
    prediction = predict_with_sagemaker(predictor, image_data, text_data)
    
    print(prediction)

    # Clean up
    deployer.delete_endpoint(predictor)

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