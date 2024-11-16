from datetime import datetime
import json
import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from extra.constants import OUTPUT_PATH
from extra.dalle.dalle_image_generation import generate_posts_dalle
from extra.replicate.replicate_models_call import generate_posts_replicate_model
from extra.unsplash.post_geneator import generate_posts as generate_posts_unsplash

load_dotenv(override=True)
st.title("{po.è.me} Playground - Geração de Posts")

# Initialize session state variables
if "selected_method" not in st.session_state:
    st.session_state.selected_method = "Auto (Unsplash + IA)"
if "status_postagem" not in st.session_state:
    st.session_state.status_postagem = 'nao_gerada'
if "descricao_postagem" not in st.session_state:
    st.session_state.descricao_postagem = ''
if "legenda_postagem" not in st.session_state:
    st.session_state.legenda_postagem = ''
if "n_postagens" not in st.session_state:
    st.session_state.n_postagens = 5
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []
if "query_used" not in st.session_state:
    st.session_state.query_used = ''
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def authenticate(password):
    if password == os.getenv("PASSWORD"):
        st.session_state.authenticated = True
    else:
        st.error("Password incorrect")


if not st.session_state.authenticated:
    st.header("Please enter the password to access the app")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        authenticate(password)
else:
    # Sidebar menu for selecting image generation method
    with st.sidebar:
        st.header("Configurações")
        st.session_state.selected_method = st.selectbox(
            "Geração de imagens preferencialmente via:", 
            options=[
                "Ideogram V2 (Recomendado)",
                "Recraft V3 Images",
                "Flux Pro",
                "Flux Pro Ultra",
                "Recraft V3 SVG (para Logos)",
                #"Auto (Unsplash + IA)",
                "Unsplash (com validação de vision)",
                "Unsplash (direto sem validação)",
            ]
        )

        if st.session_state.selected_method in ["Ideogram V2 (Recomendado)", "Recraft V3 Images"]:
            st.session_state.caption_type = st.radio(
                "Tipo de Legenda:",
                options=[
                    "Legenda no próprio prompt (integrada na imagem)",
                    "Legenda inserida (após a geração da imagem)"
                ],
                index=0
            )
        else:
            st.session_state.caption_type = "Legenda inserida (após a geração da imagem)"

    def log_action(action_description, data):
        """Log actions for debugging purposes"""
        st.session_state.debug_logs.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action_description,
            "data": data
        })

    def export_debug_logs():
        """Export debug logs to a text file"""
        log_content = "\n".join(json.dumps(log, indent=2) for log in st.session_state.debug_logs)
        return log_content

    def show_generated_posts():
        files = []
        # Check if the directory exists first
        if not os.path.exists(OUTPUT_PATH):
            st.error("No posts directory found")
            return
        
        # If it's a direct file in the output path (from replicate models)
        if any(file.startswith('alternativa_') for file in os.listdir(OUTPUT_PATH)):
            files = [os.path.join(OUTPUT_PATH, f) for f in os.listdir(OUTPUT_PATH)]
        else:
            # Original logic for nested structure (Unsplash)
            for alternative in os.listdir(OUTPUT_PATH):
                altpath = os.path.join(OUTPUT_PATH, alternative)
                if os.path.isdir(altpath) and os.listdir(altpath):  # Check if it's a directory and not empty
                    filepath = os.path.join(altpath, os.listdir(altpath)[0])
                    files.append(filepath)

        if len(files) > 5:
            col1, col2 = st.columns(2)
            with col1:
                for file in files[:5]:
                    st.image(file)
            with col2:
                for file in files[5:]:
                    st.image(file)
        else:
            for file in files:
                st.image(file)

    def reset_post_status():
        st.session_state['status_postagem'] = 'nao_gerada'
        st.session_state['legenda_postagem'] = ''  # Clear the caption
        st.session_state['descricao_postagem'] = ''  # It's good practice to clear this too

    def execute_generate_posts():
        if not all([st.session_state['descricao_postagem'], st.session_state['n_postagens']]):
            st.error('Algum erro inesperado ocorreu, por favor tente novamente')
        else:
            st.session_state['status_postagem'] = 'gerando'

    def generate_posts_from_user_input(descricao, legenda, n_exemplos, posicao):
        result_posts_filepath = os.path.join('.', 'temp', 'posts')
        if os.path.exists(result_posts_filepath):
            shutil.rmtree(result_posts_filepath)

        result_images_filepath = os.path.join('.', 'temp', 'images')
        if os.path.exists(result_images_filepath):
            shutil.rmtree(result_images_filepath)

        query_used = ""
        num_images = 0

        if st.session_state.selected_method == "Unsplash (direto sem validação)":
            query_used, num_images = generate_posts_unsplash(
                image_description=descricao, 
                image_caption=legenda, 
                images_sample=n_exemplos, 
                vision_validate=False,
                caption_position=posicao
            )
        elif st.session_state.selected_method == "Unsplash (com validação de vision)":
            query_used, num_images = generate_posts_unsplash(
                image_description=descricao, 
                image_caption=legenda, 
                images_sample=n_exemplos,
                caption_position=posicao
            )
        else:
            # For replicate models
            model_name = ""
            if "Recraft V3 SVG" in st.session_state.selected_method:
                model_name = "recraft-ai/recraft-v3-svg"
            elif "Flux Pro Ultra" in st.session_state.selected_method:
                model_name = "black-forest-labs/flux-1.1-pro-ultra"
            elif "Flux Pro" in st.session_state.selected_method:
                model_name = "black-forest-labs/flux-1.1-pro"
            elif "Ideogram V2" in st.session_state.selected_method:
                model_name = "ideogram-ai/ideogram-v2"
            elif "Recraft V3 Images" in st.session_state.selected_method:
                model_name = "recraft-ai/recraft-v3"

            # Determine if caption should be inserted via prompt
            insert_caption_via_prompt = False
            if st.session_state.selected_method in ["Ideogram V2 (Recomendado)", "Recraft V3 Images"]:
                if st.session_state.caption_type == "Legenda no próprio prompt (integrada na imagem)":
                    insert_caption_via_prompt = True
                else:
                    insert_caption_via_prompt = False
            else:
                insert_caption_via_prompt = False  # For other models, always use image_caption.py

            query_used = generate_posts_replicate_model(
                image_description=descricao, 
                image_caption=legenda, 
                model=model_name,
                images_sample=n_exemplos,
                caption_position=posicao,
                insert_caption_via_prompt=insert_caption_via_prompt,
                style="realistic_image" if model_name == "recraft-ai/recraft-v3" else None  # Add style parameter
            )

        st.session_state.query_used = query_used

    # Main content
    st.header('Geração de Postagens')

    if st.session_state['status_postagem'] == 'nao_gerada':
        st.write('Gere postagens para publicar em suas redes sociais')

        descricao_postagem = st.text_area(label='Descrição da imagem', height=100, placeholder='Um carro preto')
        if descricao_postagem:
            st.session_state['descricao_postagem'] = descricao_postagem

        legenda_postagem = st.text_input(label='Legenda / Texto da Imagem (se houver)')
        if legenda_postagem:
            st.session_state['legenda_postagem'] = legenda_postagem

        if legenda_postagem:
            posicao_legenda = st.selectbox(
                label='Posição da Legenda',
                options=['superior', 'meio da imagem', 'inferior'],
                index=2
            )
            st.session_state['posicao_legenda'] = posicao_legenda

        n_postagens = st.selectbox(label='Numero de postagens: ', options=[i+1 for i in range(10)], index=4)
        st.session_state['n_postagens'] = n_postagens

        st.button('Gerar Postagens', on_click=execute_generate_posts)

    elif st.session_state['status_postagem'] == 'gerada':
        st.write(f"Palavras usadas na pesquisa de imagem: {st.session_state.query_used}")
        st.button('Gerar novamente', on_click=reset_post_status)
        show_generated_posts()

    elif st.session_state['status_postagem'] == 'gerando':
        with st.spinner(text='Gerando postagens. Isso pode levar alguns minutos'):
            generate_posts_from_user_input(st.session_state['descricao_postagem'],
                                        st.session_state['legenda_postagem'],
                                        st.session_state['n_postagens'],
                                        st.session_state.get('posicao_legenda', 'inferior'))

            st.session_state['status_postagem'] = 'gerada'
            st.success('Postagem gerada com sucesso')
            st.button('Visualizar')