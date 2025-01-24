import unittest
import logging
from src.dataset import Dataset
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_path = 'data/test_dataset.json'
        self.image_folder = 'data/assets/test'
        
        # Create a Dataset instance
        self.dataset = Dataset(self.dataset_path, self.image_folder)
        
        logging.info("Dataset initialized")

    def test_load_raw_data(self):
        # Test loading raw data
        raw_data = self.dataset.load_raw_data()
        logging.info("Raw data loaded")
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)

    def test_clean_data(self):
        # Test cleaning data
        cleaned_data = self.dataset.clean_data()
        logging.info("Cleaned data")
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertGreater(len(cleaned_data), 0)

    def test_load_and_embed_images(self):
        # Test loading and embedding images
        image_embeddings = self.dataset.load_and_embed_images()
        logging.info("Image embeddings generated")
        self.assertIsInstance(image_embeddings, np.ndarray)
        self.assertEqual(image_embeddings.shape[0], len(self.dataset.cleaned_data))

    def test_embed_text(self):
        # Test text embedding
        text_embeddings = self.dataset.embed_text()
        logging.info("Text embeddings generated")
        
        # Add assertions as needed
        self.assertIsNotNone(text_embeddings)
        self.assertIsInstance(text_embeddings, np.ndarray)
        self.assertEqual(text_embeddings.shape[0], len(self.dataset.cleaned_data))

    def test_embed_image(self):
        # Test image embedding
        self.dataset.clean_data()  # Ensure data is cleaned before embedding
        image_embeddings = self.dataset.load_and_embed_images()
        logging.debug("Image embeddings generated.")
        self.assertIsInstance(image_embeddings, np.ndarray)
        self.assertEqual(image_embeddings.shape[0], len(self.dataset.cleaned_data))
    
    def test_action_labels(self):
        # Test labeling actions
        action_labels = self.dataset.label_actions()
        logging.info("Action labels generated.")

        # Add assertions as needed
        self.assertIsNotNone(action_labels)
        self.assertIsInstance(action_labels, np.ndarray)
        self.assertEqual(action_labels.shape[0], len(self.dataset.cleaned_data))

if __name__ == '__main__':
    unittest.main()