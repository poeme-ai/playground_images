import os
import requests
from PIL import Image
from io import BytesIO

ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')
TEMP_IMAGES_DIR = os.path.join('.', 'temp', 'images')

def download_image(image_url, save_path):
    response = requests.get(image_url)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    image.save(save_path)

def search_images(query, max_images=5, orientation='portrait'):
    url = f'https://api.unsplash.com/search/photos/?query={query}&per_page={max_images}&orientation={orientation}&client_id={ACCESS_KEY}'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['results']