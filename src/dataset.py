import boto3
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import TextVectorization, Input, Dense, Concatenate, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO

load_dotenv(dotenv_path='.env')

class Dataset:
    
    def __init__(self, dataset_path: str = None, s3_image_bucket: str = None) -> None:
        self.s3_client = boto3.client('s3')
        self.s3_image_bucket = s3_image_bucket
        self.dataset_path = self.create_and_save_input_text() if dataset_path is None else dataset_path
        self.action_labels = None

    def load_raw_data(self):
        '''
        Loads text dataset and 
        returns it as a pandas 
        DataFrame
        '''
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def clean_data(self, raw_data: pd.DataFrame):
        '''
        Cleans the raw data, dropping some actions,
        protecting against null values, removing punctuation, 
        and handling missing turn data for image filenames.
        '''
        df = pd.DataFrame(raw_data)

        # Confirm valid "turn" value
        df['turn'] = df['turn'].apply(lambda x: x if isinstance(x, str) and x.strip() else None)
        df.dropna(subset=['turn'], inplace=True)

        # Drops all actions but the first (multi-action prediction not yet supported)
        df['actions'] = df['actions'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
        
        # Clean actions data
        df['actions'] = df['actions'].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True)

        # Clean game_state data
        df['game_state'] = df['game_state'].apply(lambda x: [str(text).lower().replace('[^\w\s]', '') for text in x])

        # Handle null or missing values in 'actions' column
        df['actions'] = df['actions'].fillna('')

        # Handle null or missing values in 'game_state' column
        df['game_state'] = df['game_state'].apply(lambda x: x if isinstance(x, list) else [])
        df['game_state'] = df['game_state'].apply(lambda x: [text if text is not None else '' for text in x])

        # Drop entries with no "screenshot" data or that aren't .jpgs
        df.dropna(subset=['screenshot'], inplace=True)
        df = df[df['screenshot'].str.endswith('.jpg')]

        # Force "screenshot" values to strings
        df['screenshot'] = df['screenshot'].astype(str)

        # Remove data/assets/ prefix from screenshot filenames
        df['screenshot'] = df['screenshot'].apply(lambda x: x.replace('data/assets/', ''))

        return df

    def list_images(self, cleaned_data: pd.DataFrame):
        '''
        Lists all the image 
        filenames from the cleaned 
        data in text dataset
        '''

        images_list = cleaned_data['screenshot'].tolist()
        
        return images_list
    
    def list_images_from_s3(self):
        '''
        Lists all the image 
        filenames from the S3 bucket
        '''

        response = self.s3_client.list_objects_v2(Bucket=self.s3_image_bucket)
        images_list = [obj['Key'] for obj in response.get('Contents', [])]
        
        return images_list


    def load_and_embed_images(self, cleaned_images_list: list):
        '''
        Downloads, handles, and embeds 
        images using ResNet50 and returns 
        the embeddings as a numpy array
        '''

        model = self._load_images_model()

        
        self._load_images_from_s3(cleaned_images_list)
        preprocessed_images = self._preprocess_images(cleaned_images_list)
        image_embeddings = self._embed_images(preprocessed_images, model)
        return image_embeddings

    def _load_images_model(self):
        '''
        Loads the ResNet model used in _embed_images
        '''
        # Load the ResNet50 model pre-trained on ImageNet
        base_model = ResNet50(weights='imagenet', include_top=False)

        # Create a new model that outputs the embeddings
        model = Model(inputs=base_model.input, outputs=base_model.output)

        return model
    
    def _load_images_from_s3(self, cleaned_images_list: list) -> np.ndarray:
        '''
        Downloads and loads images from S3
        '''
        preprocessed_images = []

        # Ensure the 'assets' directory exists
        os.makedirs('data/assets', exist_ok=True)

        for filename in cleaned_images_list:
            obj = self.s3_client.get_object(Bucket=self.s3_image_bucket, Key=filename)
            with open(f'data/assets/{filename}', 'wb') as f:
                f.write(obj['Body'].read())
            img = Image.open(f'data/assets/{filename}')

    def _preprocess_images(self, cleaned_images_list: list) -> np.ndarray:
        '''
        Helper function to load and preprocess images for ResNet50 embedding
        '''
        preprocessed_images = []
        
        for filename in cleaned_images_list:
            # Load image from local /assets directory
            img = Image.open(f'data/assets/{filename}')
            # Convert image to array
            img_array = image.img_to_array(img)
            # Expand dimensions to match the input shape of ResNet50
            img_array = np.expand_dims(img_array, axis=0)
            # Preprocess the image for ResNet50
            img_array = preprocess_input(img_array)
            preprocessed_images.append(img_array)
        
        # Convert list to numpy array
        preprocessed_images = np.vstack(preprocessed_images)
        
        return preprocessed_images

    def _embed_images(self, preprocessed_images: np.ndarray, model: Model) -> np.ndarray:
        '''
        Embeds preprocessed images using ResNet50
        '''
        
        # Assumes preprocessed_images is an array of preprocessed images
        embeddings = model.predict(preprocessed_images)

        # Flatten the embeddings
        image_embeddings = embeddings.reshape(embeddings.shape[0], -1)

        return image_embeddings

    def embed_text(self, cleaned_data: pd.DataFrame) -> np.ndarray:
        '''
        Embeds text using Keras TextVectorization
        '''

        # Initialize TextVectorization layer
        vectorizer = TextVectorization(output_mode='tf-idf')
        
        # Adapt the vectorizer to the text data
        vectorizer.adapt(cleaned_data['game_state'].apply(lambda x: ' '.join(x)))
        
        # Transform the text data
        text_embeddings = vectorizer(cleaned_data['game_state'].apply(lambda x: ' '.join(x)))
        
        text_embeddings = text_embeddings.numpy()
        
        return text_embeddings

    def label_actions(self, cleaned_data: pd.DataFrame) -> np.ndarray:
        '''
        Labels actions data
        '''

        # Assuming actions are categorical and need to be encoded
        actions = cleaned_data['actions']
        
        # Convert actions to a numerical format, e.g., using LabelEncoder
        label_encoder = LabelEncoder()
        action_labels = label_encoder.fit_transform(actions)
        
        # Store the action labels in the instance variable
        self.action_labels = action_labels
        
        return action_labels
    
    def save_label_mappings(self, filepath: str) -> None:
        '''
        Saves the label mappings to file
        '''
        with open(filepath, 'wb') as f:
            joblib.dump(self.label_encoder, f)

    def load_label_mappings(self, filepath: str) -> None:
        '''
        Loads the label mappings from file
        '''
        with open(filepath, 'rb') as f:
            self.label_encoder = joblib.load(f)
        self.action_labels = self.label_encoder.transform(self.action_labels)

    def decode_labels(self, labels: np.ndarray) -> np.ndarray:
        '''
        Decodes labels using the label encoder
        '''
        return self.label_encoder.inverse_transform(labels)
    
    def create_and_save_input_text(self) -> str:
        '''
        Generates input text for the model 
        using ChatGPT, where text data is not
        already present, for images in an S3 bucket
        '''

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Fetch the list of image URLs from the S3 bucket
        response = self.s3_client.list_objects_v2(Bucket=self.s3_image_bucket)
        image_urls = [
            f"https://{self.s3_image_bucket}.s3.amazonaws.com/{obj['Key']}"
            for obj in response.get('Contents', [])
        ]

        # Initialize a list to store the generated texts
        text_inputs = []

        class game_state(BaseModel):
            turn: str 
            science_per_turn: int
            culture_per_turn: int
            gold_per_turn: int
            faith_per_turn: int
            military_power: int
            city_count: int
            unit_count: int
            units: list[str]
            cities: list[str]
            current_research: str
            current_civic: str
            era_score: int

        # Iterate over each image URL
        for image_url in image_urls:
            # Create a completion request with the image URL
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": os.getenv("CHATGPT_SYSTEM_PROMPT")},
                    {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": os.getenv("CHATGPT_SYSTEM_PROMPT")},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                    },
                                },
                            ],
                        },
                ],
                response_format=game_state,
            )
        game_state = completion.choices[0].message.parsed


        # Add the completion text to the generated_texts list
        text_inputs.append(
            {
                "turn": game_state.turn,
                "actions": None,
                "game_state": game_state,
                "screenshot": None # really this should be the image filename but I can't be bothered
            }
        )

        dataset_path = self._save_input_text(text_inputs)

        return dataset_path
    
    def _save_input_text(self, generated_texts: list) -> None:
        '''
        Saves the generated texts to file
        '''
        
        with open('data/prediction_input.json', 'w') as f:
            json.dump(generated_texts, f)
        
        return 'data/prediction_input.json'