import os
from PIL import Image
import requests
from io import BytesIO
import replicate

from extra.utilities.image_caption import add_caption_to_image

TEMP_IMAGES_DIR = os.path.join('.', 'temp', 'images')

def generate_posts_replicate_model(image_description, image_caption, model="", images_sample=1, start_index=0):
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    successful_generations = 0

    for i in range(images_sample):
        try:
            # Start the prediction
            prediction = replicate.run(
                model,
                input={"prompt": image_description}
            )
            
            print(f'prediction: {prediction}')
            
            # The prediction is already a direct URL string
            image_url = prediction
            if not image_url:
                print(f"Warning: No valid image URL received for image {i + 1}")
                continue

            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            image_path = os.path.join(TEMP_IMAGES_DIR, f'dalle_image_{start_index + successful_generations + 1}.png')
            img.save(image_path)

            if image_caption:
                add_caption_to_image(image_path, [image_caption], start_index + successful_generations + 1)
            
            successful_generations += 1

        except Exception as e:
            print(f"Error generating image {i + 1}: {str(e)}")
            continue

    if successful_generations == 0:
        raise ValueError(f"Failed to generate any images with model {model}")

    return image_description
