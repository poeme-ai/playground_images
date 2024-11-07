import logging
import os
from extra.constants import OUTPUT_PATH, TEMP_IMAGES_DIR
from extra.utilities.check_image_vision import check_image_adequacy
from extra.utilities.image_caption import add_caption_to_image
from dotenv import load_dotenv

from extra.agent_basic_llm.llm_api import ChatApp
from extra.unsplash.get_unsplash_images import download_image, search_images

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

simple_image_description_to_prompt = """
You are an AI specialized in returning few keywords to be used in an image search on Unsplash, based on the image description you get. 
The query needs to be in English. 
Return only the keywords.

Example:
Input: "A group of friends having a picnic in the park."
Your awnser: friends picnic park
"""

def generate_posts(
        image_description:str, 
        image_caption:str, 
        post_type:str='image', 
        add_black_background:bool = True,
        images_sample:int = 10,
        vision_validate = True,
        caption_position = 'inferior'
    ):

    if post_type not in ['image', 'text']:
        raise ValueError('Incorrect value for post_type. Parameter only acept "image" or "text" ')
    
    ai_helper = ChatApp(model='gpt-4o-2024-05-13')
    image_descripto_to_query = ai_helper.single_message_completion(image_description, simple_image_description_to_prompt)

    images = search_images(image_descripto_to_query, max_images=images_sample)

    logger.info('Image search complete')

    num_returned_images = 0
    for idx, img in enumerate(images):
        image_url = img['urls']['regular']
        if vision_validate == True:
            if check_image_adequacy(image_url, image_descripto_to_query):
                if image_caption:
                    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
                    image_path = os.path.join(TEMP_IMAGES_DIR, f'image_{idx + 1}.jpg')
                    download_image(image_url, image_path)
                    num_returned_images +=1
                else:
                    os.makedirs(OUTPUT_PATH, exist_ok=True)
                    image_path = os.path.join(OUTPUT_PATH, f'alternativa_{idx + 1}.jpg')
                    download_image(image_url, image_path)
                    num_returned_images +=1
        else:
            if image_caption:
                os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
                image_path = os.path.join(TEMP_IMAGES_DIR, f'image_{idx + 1}.jpg')
                download_image(image_url, image_path)
                num_returned_images +=1
            else:
                image_url = img['urls']['regular']
                os.makedirs(OUTPUT_PATH, exist_ok=True)
                image_path = os.path.join(OUTPUT_PATH, f'alternativa_{idx + 1}.jpg')
                download_image(image_url, image_path)
                num_returned_images +=1
    
    if image_caption:
        logger.info('Generating captions ...')  
        for idx, img in enumerate(os.listdir(TEMP_IMAGES_DIR)):
            image_path = os.path.join('.', 'temp', 'images', img)
            add_caption_to_image(
                image_path, 
                [image_caption],
                idx+1,
                post_type,
                add_black_background,
                caption_position
            )

    logger.info('Image download complete')
            
    return image_descripto_to_query, num_returned_images