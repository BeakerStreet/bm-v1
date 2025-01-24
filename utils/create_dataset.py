import boto3
import json
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')

def download_all_actions(bucket_name):
    logger.info(f"Downloading all actions from bucket: {bucket_name}")
    action_keys = []
    response = s3.list_objects_v2(Bucket=bucket_name)
    for obj in response.get('Contents', []):
        action_keys.append(obj['Key'])

    actions_data = {}
    for key in action_keys:
        local_path = os.path.join('/tmp', key.split('/')[-1])
        s3.download_file(bucket_name, key, local_path)
        with open(local_path, 'r') as f:
            action_data = json.load(f)
            filename = key.split('/')[-1]
            actions_data[filename] = action_data

    return actions_data

def download_all_texts(bucket_name):
    logger.info(f"Downloading all texts from bucket: {bucket_name}")
    text_keys = []
    response = s3.list_objects_v2(Bucket=bucket_name)
    for obj in response.get('Contents', []):
        text_keys.append(obj['Key'])

    texts_data = {}
    for key in text_keys:
        local_path = os.path.join('/tmp', key.split('/')[-1])
        s3.download_file(bucket_name, key, local_path)
        with open(local_path, 'r') as f:
            text_data = json.load(f)
            filename = key.split('/')[-1]
            texts_data[filename] = text_data

    return texts_data

def download_all_images(bucket_name, output_folder, prefix=''):
    logger.info(f"Downloading all images from bucket: {bucket_name} with prefix: {prefix}")
    image_keys = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response.get('Contents', []):
        image_keys.append(obj['Key'])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output folder: {output_folder}")

    images_data = {}
    for key in image_keys:
        local_path = os.path.join(output_folder, key.split('/')[-1])
        s3.download_file(bucket_name, key, local_path)
        filename = key.split('/')[-1]
        images_data[filename] = local_path

    return images_data

def load_local_images(output_folder):
    logger.info(f"Loading local images from folder: {output_folder}")
    images_data = {}
    for filename in os.listdir(output_folder):
        if filename.endswith('.jpg'):
            local_path = os.path.join(output_folder, filename)
            images_data[filename] = local_path
    return images_data

def create_combined_dataset(actions_data, texts_data, images_data, output_file):
    logger.info("Creating combined dataset")

    combined_data = []
    for filename, action_data in actions_data.items():
        root_filename = filename.replace('-action.json', '')
        text_filename = f"{root_filename}-text.json"
        image_filename = f"{root_filename}-image.jpg"
        logger.info(f"Checking root ({root_filename}), {text_filename} ({type(text_filename)}) and {image_filename} ({type(image_filename)})")
        if text_filename in texts_data and image_filename in images_data:
            combined_entry = {
                'turn': root_filename,
                'actions': action_data['actions_by_player'],
                'game_state': texts_data[text_filename],
                'screenshot': images_data[image_filename]
            }
            combined_data.append(combined_entry)
            logger.info(f"Combined data for file: {filename}")

    with open(output_file, 'w') as f:
        json.dump(combined_data, f)
    logger.info(f"Combined dataset saved to {output_file}")

# Download actions, texts, and optionally images
actions_bucket_name = 'bm-v1-training-actions-json'
texts_bucket_name = 'bm-v1-training-text-json'
images_bucket_name = 'bm-v1-training-images'
output_folder = 'data/assets'

actions_data = download_all_actions(actions_bucket_name)
texts_data = download_all_texts(texts_bucket_name)

# Check if images download is enabled
download_images = '--download-images' in sys.argv

if download_images:
    images_data = download_all_images(images_bucket_name, output_folder)
else:
    images_data = load_local_images('data/assets')

# Create combined dataset
output_file = 'data/dataset.json'
create_combined_dataset(actions_data, texts_data, images_data, output_file)