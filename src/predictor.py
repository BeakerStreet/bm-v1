import numpy as np
import boto3
import json

class Predict:
    def __init__(self, model_url: str, image_input: list, text_input: list): # text and image inputs expected as lists of np.ndarrays
        
        self.model_url = model_url
        self.image_input = image_input
        self.text_input = text_input
        self.runtime_client = boto3.client('sagemaker-runtime')

    def make_prediction(self):

        # Prepare the payload for the SageMaker endpoint
        payload = {
            'instances': [
                {'image_input': self.image_input, 'text_input': self.text_input}
            ]
        }

        # Make a prediction
        response = self.runtime_client.invoke_endpoint(
            EndpointName=self.model_url,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        prediction = json.loads(response['Body'].read().decode())
        print(f"Prediction: {prediction}")

        return prediction