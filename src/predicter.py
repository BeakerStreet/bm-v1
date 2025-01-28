import numpy as np
from tensorflow.keras.models import load_model
from deployer import Deploy
from src.dataset import Dataset

class Predict:
    def __init__(self, dataset_path: str, image_folder: str):
        # Load dataset
        self.dataset = Dataset(dataset_path=dataset_path, image_folder=image_folder)
        
        # Initialize, upload, and deploy
        self.deployer = Deploy()

    def make_prediction(self):
        # Prepare the data for prediction
        image_data = self.dataset.image_embeddings
        text_data = self.dataset.text_embeddings
        
        # Make a prediction
        # Note: Adjust this line as needed to match the actual prediction call
        prediction = self.deployer.url.predict([image_data, text_data])
        
        print(f"Prediction: {prediction}")
        
        # Clean up
        self.deployer.delete_endpoint(self.deployer.url)

# Example usage:
# predictor = Predict(dataset_path='data/prediction_input.json', image_folder='data/assets/predict')
# predictor.make_prediction() 