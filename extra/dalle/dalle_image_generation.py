import os
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

from extra.utilities.image_caption import add_caption_to_image

TEMP_IMAGES_DIR = os.path.join('.', 'temp', 'images')

def generate_posts_dalle(image_description, image_caption, images_sample=1, start_index=0):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

    for i in range(images_sample):
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_description,
            n=1,  # Generate one image at a time
            size="1024x1024",
            quality="standard",
        )

        image_url = response.data[0].url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        image_path = os.path.join(TEMP_IMAGES_DIR, f'dalle_image_{start_index + i + 1}.png')
        img.save(image_path)

        if image_caption:
            add_caption_to_image(image_path, [image_caption], start_index + i + 1)

    return image_description
