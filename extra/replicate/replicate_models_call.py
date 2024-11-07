import os
from PIL import Image
import requests
from io import BytesIO
import replicate

from extra.constants import OUTPUT_PATH, TEMP_IMAGES_DIR
from extra.utilities.image_caption import add_caption_to_image

def generate_posts_replicate_model(
    image_description,
    image_caption,
    model="",
    images_sample=1,
    start_index=0,
    caption_position='inferior',
    insert_caption_via_prompt=True,
    style=None
):
    successful_generations = 0

    # Modify prompt based on caption if insert_caption_via_prompt is True
    prompt = image_description
    if image_caption and insert_caption_via_prompt:
        position_text = {
            'superior': 'na parte de cima da imagem',
            'meio da imagem': 'no centro da imagem, de forma sobreposta',
            'inferior': 'na parte de baixo da imagem'
        }
        prompt = f"{image_description}. A imagem deve conter uma legenda com exatamente o seguinte texto: \"{image_caption}\". A legenda deve estar {position_text[caption_position]}"

    for i in range(images_sample):
        try:
            # Start the prediction with modified prompt
            input_params = {"prompt": prompt}
            if style:
                input_params["style"] = style

            prediction = replicate.run(
                model,
                input=input_params
            )

            # The prediction may return a list or a single URL
            if isinstance(prediction, list):
                image_url = prediction[0]
            else:
                image_url = prediction

            if not image_url:
                print(f"Warning: No valid image URL received for image {i + 1}")
                continue

            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            if not insert_caption_via_prompt:
                os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
                image_path = os.path.join(TEMP_IMAGES_DIR, f'dalle_image_{start_index + successful_generations + 1}.png')
            else:
                os.makedirs(OUTPUT_PATH, exist_ok=True)
                image_path = os.path.join(OUTPUT_PATH, f'alternativa_{start_index + successful_generations + 1}.png')
            img.save(image_path)

            # If insert_caption_via_prompt is False, add caption using image_caption.py
            if image_caption and not insert_caption_via_prompt:
                add_caption_to_image(
                    image_path,
                    [image_caption],
                    start_index + successful_generations + 1,
                    post_type='image',
                    add_black_background=True,
                    caption_position=caption_position
                )

            successful_generations += 1

        except Exception as e:
            print(f"Error generating image {i + 1}: {str(e)}")
            continue

    if successful_generations == 0:
        raise ValueError(f"Failed to generate any images with model {model}")

    return prompt
