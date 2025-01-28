import numpy as np
import boto3

class Predict:
    def __init__(self, model_url: str, image_input: np.ndarray, text_input: np.ndarray): # text and image inputs expected as embeddings
        self.model_url = model_url
        self.image_input = image_input
        self.text_input = text_input

    def make_prediction(self):
        image_data = self.image_input
        text_data = self.text_input

        # Make a prediction
        prediction = self.model_url.predict([image_data, text_data])
        print(f"Prediction: {prediction}")

        return prediction