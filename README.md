## Playground Images
Contém o código para criação de Playground de teste de diferentes métodos de geração de imagens, com legegenda opcional, utilizando Streamlit, com intuito de permitir testes para definição da estratégia de geração de Posts da Poème

## Utilização Local - Instalação via git clone:
Requisitos: instalação de git e python.

```bash
git clone https://github.com/poeme-ai/playground_images.git
python -m venv .venv
```
### Windows:
```bash
./.venv/Scripts/Activate
```
### Mac:
```bash
source .venv/bin/activate
```


### Instalar as dependências:
```bash
pip install -r requirements.txt
```

### Atribuir variáveis de ambiente:
Criar arquivo .env:
```bash
OPENAI_API_KEY="<chave aqui>"
REPLICATE_API_TOKEN="<chave aqui>"
UNSPLASH_APP_ID="<chave aqui>"
UNSPLASH_ACCESS_KEY="<chave aqui>"
UNSPLASH_SECRET_KEY="<chave aqui>"
```

### Iniciar aplicação:
```bash
streamlit run app.py
```
