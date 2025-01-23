import logging
import boto3
import numpy as np
from dataset import Dataset
from tensorflow.keras.models import load_model
import sys

s3 = boto3.client('s3')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

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

    # Load the model
    model = load_model('model.keras')

    # Load the dataset for prediction
    dataset = Dataset(dataset_path='data/dataset.json', image_folder='data/assets/predict')

    # Select a random image and its corresponding text embedding
    random_index = np.random.randint(len(dataset.image_embeddings))
    image_data = np.expand_dims(dataset.image_embeddings[random_index], axis=0)
    text_data = np.expand_dims(dataset.text_embeddings[random_index], axis=0)

    # Make a prediction
    prediction = model.predict([image_data, text_data])
    
    print(prediction)

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