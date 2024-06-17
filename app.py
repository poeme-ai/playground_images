from datetime import datetime
import json
import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from extras.unsplash.post_geneator import generate_posts

load_dotenv()
st.title("{po.è.me} Playground - Post Generation")

# Initialize session state variables
if "selected_method" not in st.session_state:
    st.session_state.selected_method = "Unsplash"
if "status_postagem" not in st.session_state:
    st.session_state.status_postagem = 'nao_gerada'
if "descricao_postagem" not in st.session_state:
    st.session_state.descricao_postagem = ''
if "legenda_postagem" not in st.session_state:
    st.session_state.legenda_postagem = ''
if "n_postagens" not in st.session_state:
    st.session_state.n_postagens = 1
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# Sidebar menu for selecting image generation method
with st.sidebar:
    st.header("Configurações")
    st.session_state.selected_method = st.selectbox("Geração de imagens via:", options=["Unsplash", "Other Method"])

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

def get_generated_slides():
    return open('./temp/output.pptx', 'rb')

def show_generated_posts():
    files = []
    if not st.session_state['legenda_postagem']:
        posts_dir = os.path.join('.', 'temp', 'images')
        files = [os.path.join(posts_dir, file) for file in os.listdir(posts_dir)]
    else:
        posts_dir = os.path.join('.', 'temp', 'posts')
        for alternative in os.listdir(posts_dir):
            altpath = os.path.join(posts_dir, alternative)
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

def execute_generate_posts():
    if not all([st.session_state['descricao_postagem'], st.session_state['n_postagens']]):
        st.error('Algum erro inesperado ocorreu, por favor tente novamente')
    else:
        st.session_state['status_postagem'] = 'gerando'

def generate_posts_from_user_input(descricao, legenda, n_exemplos):
    result_posts_filepath = os.path.join('.', 'temp', 'posts')
    if os.path.exists(result_posts_filepath):
        shutil.rmtree(result_posts_filepath)

    result_images_filepath = os.path.join('.', 'temp', 'images')
    if os.path.exists(result_images_filepath):
        shutil.rmtree(result_images_filepath)

    generate_posts(image_description=descricao, image_caption=legenda, images_sample=n_exemplos)

# Main content
st.header('Geração de Postagens')

if st.session_state['status_postagem'] == 'nao_gerada':
    st.write('Gere postagens para publicar em suas redes sociais')

    descricao_postagem = st.text_area(label='Descrição da imagem', height=100, placeholder='Uma foto temática de um recém-nascido com destaque para o teste do pezinho, com ícones médicos e gráficas relacionadas ao exame.')
    if descricao_postagem:
        st.session_state['descricao_postagem'] = descricao_postagem

    legenda_postagem = st.text_input(label='Legenda para adicionar na imagem', placeholder='Opicional')
    if legenda_postagem:
        st.session_state['legenda_postagem'] = legenda_postagem

    n_postagens = st.selectbox(label='Numero de postagens: ', options=[i+1 for i in range(10)])
    st.session_state['n_postagens'] = n_postagens

    st.button('Gerar Postagens', on_click=execute_generate_posts)

elif st.session_state['status_postagem'] == 'gerada':
    st.button('Gerar novamente', on_click=reset_post_status)
    show_generated_posts()

elif st.session_state['status_postagem'] == 'gerando':
    with st.spinner(text='Gerando postagens. Isso pode levar alguns minutos'):
        generate_posts_from_user_input(st.session_state['descricao_postagem'],
                                       st.session_state['legenda_postagem'],
                                       st.session_state['n_postagens'])

        st.session_state['status_postagem'] = 'gerada'
        st.success('Postagem gerada com sucesso')
        st.button('Visualizar')
