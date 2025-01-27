import logging
import boto3
import numpy as np
from tensorflow.keras.models import load_model
import sys
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

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

def predict():
    logger = setup_logging()
    logger.info("=== Predicting ===")

    # Load dataset
    dataset = Dataset(dataset_path='data/dataset.json', image_folder='data/assets')

    # Initialize, upload, and deploy
    logger.info("Initializing deployment, uploading and deploying model")
    model = Deploy()

    # Make a prediction
    logger.info("Predicting")
    prediction = model.url.predict([image_data, text_data])  # Adjust this line as needed
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