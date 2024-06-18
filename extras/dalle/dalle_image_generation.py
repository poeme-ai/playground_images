import os
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

from extras.utilities.image_caption import add_caption_to_image

TEMP_IMAGES_DIR = os.path.join('.', 'temp', 'images')

def generate_posts_dalle(image_description, image_caption, images_sample=1):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.images.generate(
        model="dall-e-3",
        prompt=image_description,
        n=images_sample,
        size="1024x1024",
        quality="standard",
    )

    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    for idx, image in enumerate(response.data):
        image_url = image.url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        image_path = os.path.join(TEMP_IMAGES_DIR, f'dalle_image_{idx + 1}.png')
        img.save(image_path)

        if image_caption:
            add_caption_to_image(image_path, [image_caption], idx + 1)

    return image_description
