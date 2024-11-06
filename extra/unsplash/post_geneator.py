import logging
import os
from extra.utilities.check_image_vision import check_image_adequacy
from extra.utilities.image_caption import add_caption_to_image
from dotenv import load_dotenv

from extra.agent_basic_llm.llm_api import ChatApp
from extra.unsplash.get_unsplash_images import download_image, search_images

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_IMAGES_DIR = os.path.join('.', 'temp', 'images')

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
    
    ai_helper = ChatApp(model='gpt-4-turbo')
    image_descripto_to_query = ai_helper.single_message_completion(image_description, simple_image_description_to_prompt)

    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    images = search_images(image_descripto_to_query, max_images=images_sample)

    logger.info('Image search complete')

    num_returned_images = 0
    for idx, img in enumerate(images):
        image_url = img['urls']['regular']
        if vision_validate == True:
            if check_image_adequacy(image_url, image_descripto_to_query):
                image_path = os.path.join(TEMP_IMAGES_DIR, f'image_{idx + 1}.jpg')
                download_image(image_url, image_path)
                num_returned_images +=1
        else:
            image_url = img['urls']['regular']
            image_path = os.path.join(TEMP_IMAGES_DIR, f'image_{idx + 1}.jpg')
            download_image(image_url, image_path)
            num_returned_images +=1
    
    logger.info('Image download complete')

    if image_caption:
        logger.info('Generating captions ...')  
        for idx, img in enumerate(os.listdir(TEMP_IMAGES_DIR)):
            image_path = os.path.join('.', 'temp', 'images', img)
            add_caption_to_image(
                image_path, 
                [image_caption],
                {idx+1},
                post_type,
                add_black_background,
                caption_position
            )
            
    return image_descripto_to_query, num_returned_images