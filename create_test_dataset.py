import json
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_dataset(input_file, output_file, num_rows=5):
    logger.info(f"Loading dataset from {input_file}")
    
    # Load the full dataset
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    # Select a subset of the dataset
    test_dataset = dataset[:num_rows]
    logger.info(f"Selected {len(test_dataset)} rows for the test dataset")
    
    # Save the test dataset
    with open(output_file, 'w') as f:
        json.dump(test_dataset, f)
    logger.info(f"Test dataset saved to {output_file}")

# Define file paths
input_file = 'data/dataset.json'
output_file = 'data/test_dataset.json'

# Create test dataset
create_test_dataset(input_file, output_file)