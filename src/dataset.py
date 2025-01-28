import boto3
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import TextVectorization, Input, Dense, Concatenate, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

s3 = boto3.client('s3')

class Dataset:
    
    def __init__(self, dataset_path: str, image_folder: str = None) -> None:
        self.dataset_path = dataset_path
        self.image_folder = image_folder 
        self.action_labels = None  # Initialize action_labels attribute

    def load_raw_data(self):
        '''
        Loads raw data from the dataset
        '''
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    def load_raw_images_list(self, raw_data: pd.DataFrame):
        '''
        Loads all image filenames from the image_folder
        (.jpg only)
        '''
        if not self.image_folder:
            return []

        images_list = self.raw_data['screenshot'].tolist()

        return images_list

    def clean_data(self, raw_data: pd.DataFrame):
        '''
        Cleans the raw data, dropping some actions,
        protecting against null values, removing punctuation, 
        and handling missing turn data for image filenames.
        '''
        df = pd.DataFrame(self.raw_data)

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

    def list_cleaned_images(self):
        '''
        Lists all cleaned image filenames
        '''

        images_list = self.cleaned_data['screenshot'].tolist()
        
        return images_list


    def load_and_embed_images(self):
        '''
        Loads and embeds images using ResNet50
        '''

        model = self._load_images_model()

        # Preprocess all images at once to avoid creating tf.function in a loop
        preprocessed_images = self._preprocess_images()
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

    def _preprocess_images(self) -> np.ndarray:
        '''
        Helper function to load and preprocess images for ResNet50 embedding
        '''
        preprocessed_images = []
        
        # Preprocess all images at once
        image_paths = [os.path.join(self.image_folder, filename) for filename in self.cleaned_images_list]
        for image_path in image_paths:
            # Load image
            img = image.load_img(image_path, target_size=(224, 224))
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

    def embed_text(self) -> np.ndarray:
        '''
        Embeds text using Keras TextVectorization
        '''

        # Initialize TextVectorization layer
        vectorizer = TextVectorization(output_mode='tf-idf')
        
        # Adapt the vectorizer to the text data
        vectorizer.adapt(self.cleaned_data['game_state'].apply(lambda x: ' '.join(x)))
        
        # Transform the text data
        text_embeddings = vectorizer(self.cleaned_data['game_state'].apply(lambda x: ' '.join(x)))
        
        text_embeddings = text_embeddings.numpy()
        
        return text_embeddings

    def label_actions(self) -> np.ndarray:
        '''
        Labels actions data
        '''

        # Assuming actions are categorical and need to be encoded
        actions = self.cleaned_data['actions']
        
        # Convert actions to a numerical format, e.g., using LabelEncoder
        label_encoder = LabelEncoder()
        action_labels = label_encoder.fit_transform(actions)
        
        # Store the action labels in the instance variable
        self.action_labels = action_labels
        
        return action_labels