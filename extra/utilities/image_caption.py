import os
from PIL import Image, ImageDraw, ImageFont
import random

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_caption_to_image(img_path: str, captions: 'list[str]', option_idx,
                         post_type: str = 'image', add_black_background: bool = True,
                         caption_position: str = 'inferior') -> None:
    """
    Takes an image and edits it. Draw the provided caption over the image. If add_black_background is True,
    a black square is drawn behind the text. post_type specifies the type of post. The text is centralized for a text post
    and at the bottom of the image if it is a post of type image.

    :param img_path: path to the image file
    :param captions: list of captions to be added to the image
    :param option_idx: auxiliary variable to organize the files (saves the image in alternativa_{option_idx})
    :param post_type: 'image' or 'text', the type of post that is being created
    :param add_black_background: if there is or not a black square behind the text
    :param caption_position: position of the caption on the image
    """

    # basic verifications
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'Unable to find {img_path}!')

    output_dir = os.path.join('.', 'temp', 'posts', f'alternativa_{option_idx}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    font_files = [os.path.join(font_dir, fontfile) for fontfile in os.listdir(font_dir)]
    font_path = font_files[random.randint(0, len(font_files) - 1)]

    for idx, caption in enumerate(captions):

        result_caption = caption

        n_lines = 1
        caption_words = caption.split(' ')
        n_words = len(caption_words)

        # create a multiline caption if necessary.
        if n_words > 5:
            has_rest = 0
            lines = []

            if n_words % 5 != 0:
                has_rest = 1

            n_lines = n_words // 5 + has_rest

            result_caption = ''

            # add a \n in the caption, every 5 words.
            groups_of_five_words = n_words // 5
            i = 0
            for _ in range(groups_of_five_words):
                line_words = caption_words[i:i + 5]
                line = ' '.join(line_words)
                lines.append(line)
                i += 5

            # append the last elements
            lines.append(' '.join(caption_words[i:]))

            # result caption is the final caption with the line breaks
            result_caption = '\n'.join(lines)

        # gerar mais de uma imagem se tiver mais de uma legenda
        original_name = os.path.splitext(os.path.basename(img_path))[0]

        # output file
        filename = f'{original_name}_{idx}.png'

        # get the image
        image = Image.open(img_path)
        image_width = image.size[0]
        draw = ImageDraw.Draw(image)

        # text configuration
        text_color = (255, 255, 255)  # white
        font_size = round(0.1 * image_width)
        font = ImageFont.truetype(font_path, font_size)

        # Se há mais de uma linha é necessário dimensionar a fonte com base
        # no tamanho ocupado pela maior das linhas
        if n_lines > 1:
            biggest_line = max(lines, key=len)
            text_size = font.getlength(biggest_line)
        else:
            text_size = font.getlength(result_caption.split('\n')[0])

        # number of lines
        text_box = font.getbbox(text=result_caption, stroke_width=0)
        text_height: int = n_lines * abs(text_box[1] - text_box[3])

        # define minimum and maximum size for the text
        min_text_size: float = 0.8 * image_width
        max_text_size: float = 0.9 * image_width

        n_iterations: int = 0
        while text_size >= max_text_size or text_size < min_text_size or n_iterations > 100_000:
            n_iterations += 1

            # fonte aumenta ou diminui dependendo do tamanho total do texto
            if text_size >= max_text_size:
                font_size -= 1
            else:
                font_size += 1

            # calculate the new text_size
            font = ImageFont.truetype(font_path, font_size)

            if n_lines > 1:
                # Nivelo o tamanho da postagem pelo tamanho da maior linha.
                biggest_line = max(lines, key=len)
                text_size = font.getlength(biggest_line)
            else:
                text_size = font.getlength(result_caption.split('\n')[0])

            text_box = font.getbbox(text=result_caption, stroke_width=0)
            text_height: int = n_lines * abs(text_box[1] - text_box[3])

        if n_iterations > 100_000:
            logger.info('Unable to find correct size of the text in the image. \
                         Generating with the last tried value')

        # position is a function of the post_style
        if post_type == 'image':
            if caption_position == 'superior':
                position = ((image_width - text_size) / 2, text_height)
            elif caption_position == 'meio da imagem':
                position = ((image_width - text_size) / 2, (image.size[1] - text_height) / 2)
            else:  # inferior (default)
                position = ((image_width - text_size) / 2, image.size[1] - 1.5 * text_height)
        elif post_type == 'text':
            position = ((image_width - text_size) / 2, (image_width - text_height) / 2)
        else:
            raise Exception(f'post_type={post_type} not supported!')

        if add_black_background:
            # draw a black rectangle behind the text.
            black_rec_x0y0 = (position[0] - 10, position[1] - 20)
            black_rec_x1y1 = (position[0] + text_size + 20, position[1] + text_height + 40)

            draw.rectangle(xy=[black_rec_x0y0, black_rec_x1y1], fill=(0, 0, 0))

        draw.text(position, text=result_caption, fill=text_color,
                  font=font, align='center', stroke_width=0)

        image.save(f'{output_dir}/{filename}')