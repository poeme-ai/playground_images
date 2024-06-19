import os
from extra.agent_basic_llm.llm_api import ChatApp
from openai import OpenAI

check_image_adequacy_prompt = 'You are an expert in determining if an image, based on its description, has all the elements requested. You will receive the text of an image description and the elements requested, and you should respond exactly and only with "yes" if the image contas at least two of the elemets or "no" otherwise'

def describe_image(image_url:str)->str:
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=300,
        )

    return response.choices[0].message.content

def check_image_adequacy(image_url:str, image_request:str)->bool:
    image_description = describe_image(image_url)
    ai_helper = ChatApp(model='gpt-4o-2024-05-13')
    adequacy = ai_helper.single_message_completion(
        f'Image description: {image_description}\n\nImage Request: {image_request}', 
        check_image_adequacy_prompt
    )

    print('Descrição da Imagem:', image_description)
    print('Solicitação:', image_request)
    print('Julgamento:', adequacy)


    if 'yes' in adequacy.lower() or 'sim' in adequacy.lower():
        return True
    
    return False
    
