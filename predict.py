import logging
import boto3
import numpy as np
from tensorflow.keras.models import load_model
import sys
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

from src.deploy import Deploy
from src.dataset import Dataset

def setup_logging(): # do we want this? Or should it be in the deploy.py file?
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

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

    # Load dataset
    dataset = Dataset(dataset_path='data/dataset.json', image_folder='data/assets')

    # Initialize Deploy
    deployer = Deploy()

    # Upload model to S3 using Deploy class method
    deployer.s3_upload()

    # Deploy model to SageMaker using Deploy class method
    predictor = deployer.deploy()

    # Make a prediction
    logger.info("Invoking SageMaker endpoint for prediction")
    prediction = predictor.predict([image_data, text_data])
    logger.info(f"Prediction: {prediction}")
    
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