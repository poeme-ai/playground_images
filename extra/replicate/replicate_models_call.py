import os
from PIL import Image
import requests
from io import BytesIO
import replicate

from extra.utilities.image_caption import add_caption_to_image

TEMP_IMAGES_DIR = os.path.join('.', 'temp', 'images')

def generate_posts_replicate_model(image_description, image_caption, model="", images_sample=1, start_index=0):
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

    for i in range(images_sample):
        output = replicate.run(
            model,
            input={"prompt": image_description}
        )

        response = requests.get(output[0])
        img = Image.open(BytesIO(response.content))
        image_path = os.path.join(TEMP_IMAGES_DIR, f'dalle_image_{start_index + i + 1}.png')
        img.save(image_path)

        if image_caption:
            add_caption_to_image(image_path, [image_caption], start_index + i + 1)

    return image_description
