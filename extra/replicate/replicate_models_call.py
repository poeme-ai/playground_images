import os
from PIL import Image
import requests
from io import BytesIO
import replicate

from extra.utilities.image_caption import add_caption_to_image

TEMP_IMAGES_DIR = os.path.join('.', 'temp', 'images')

def generate_posts_replicate_model(image_description, image_caption, model="", images_sample=1, start_index=0, caption_position='inferior'):
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    successful_generations = 0

    # Modify prompt based on caption
    prompt = image_description
    if image_caption:
        position_text = {
            'superior': 'acima',
            'meio da imagem': 'ao meio',
            'inferior': 'abaixo'
        }
        prompt = f"{image_description}, {position_text[caption_position]} a imagem contem como legenda o exato seguinte texto \"{image_caption}\""

    for i in range(images_sample):
        try:
            # Start the prediction with modified prompt
            prediction = replicate.run(
                model,
                input={"prompt": prompt}
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
            
            successful_generations += 1

        except Exception as e:
            print(f"Error generating image {i + 1}: {str(e)}")
            continue

    if successful_generations == 0:
        raise ValueError(f"Failed to generate any images with model {model}")

    return image_description
