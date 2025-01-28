import logging
import boto3
import numpy as np
from tensorflow.keras.models import load_model
import sys
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

from deployer import Deploy
from src.dataset import Dataset
from src.predict import Predict

def setup_logging(): 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def train():
    '''
    Trains the model with a dataset hosted
    locally in /data containing a dataset.json
    text input file and images in data/assets 
    utilising the Dataset class defined in 
    src/dataset.py
    '''

    logger = setup_logging()
    logger.info("=== Training ===")

    # Load the dataset
    dataset = Dataset(dataset_path='data/dataset.json', image_folder='data/assets')

    # Build the model
    input_shape_image = dataset.image_embeddings.shape[1:]  # Image embeddings input shape
    input_shape_text = dataset.text_embeddings.shape[1:]  # Text input shape
    num_classes = len(np.unique(dataset.labels))

    # Define the image input
    image_input = Input(shape=input_shape_image, name='image_input')
    x = Dense(1024, activation='relu')(image_input)

    # Define the text input
    text_input = Input(shape=input_shape_text, name='text_input')
    y = Dense(512, activation='relu')(text_input)

    # Concatenate the outputs
    combined = Concatenate()([x, y])
    z = Dense(num_classes, activation='softmax')(combined)

    # Create the model
    model = Model(inputs=[image_input, text_input], outputs=z)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([dataset.image_embeddings, dataset.text_embeddings], dataset.labels, epochs=10, batch_size=32)

    # Save the model
    model.save('model.keras')

    logger.info("=== Training Complete ===")

def deploy():
    '''
    Uploaded a trained model to Amazon s3 
    and then deploys to AWS SageMaker
    utilising the Deploy class defined in
    src/deploy.py
    '''

    logger = setup_logging()
    logger.info("=== Deploying ===")  

    # Initialise the Deploy class with the local model.keras file
    deployer = Deploy(model_path='model.keras')
    
    # Upload the model to S3
    s3_url = deployer.s3_upload()
    logger.info(f"Model uploaded to S3: {s3_url}")
    
    # Deploy the model to SageMaker
    endpoint_url = deployer.deploy_model()
    logger.info(f"Model deployed at endpoint: {endpoint_url}")

def predict():
    '''
    Predicts one of a couple hundred 
    predefined actions labels given a unique image 
    and a generated text input
    utilising the Predict class defined in
    src/predict.py
    '''

    # Initialize the Dataset class with the path to the dataset and image folder
    dataset = Dataset(dataset_path='data/prediction_input.json', image_folder='data/assets/predict')

    # Create an instance of the Predict class
    predictor = Predict(dataset_path='data/prediction_input.json', image_folder='data/assets/predict')
    # Make a prediction
    predictor.make_prediction()

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <train|predict|deploy>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        train()
    elif command == "predict":
        predict()
    elif command == "deploy":
        deploy()
    else:
        print("Invalid command. Use 'train' or 'predict' or 'deploy'.")
        sys.exit(1)

if __name__ == "__main__":
    main()