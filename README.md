# MBA - Busca Semântica com RAG

Um projeto de busca semântica e geração aumentada por recuperação (RAG) desenvolvido como desafio do MBA em Engenharia de Software com IA - Full Cycle.

## O que é?

Sistema que utiliza embeddings para realizar buscas semânticas em documentos PDF e gerar respostas contextualizadas através de LLMs.

**Tecnologias principais:**
- **Python** - Linguagem
- **LangChain** - Framework para aplicações com IA
- **PostgreSQL + pgVector** - Armazenamento de embeddings/vetores
- **Google Generative AI / OpenAI** - Modelos de IA

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chaves de API (Google Generative AI ou OpenAI)

## Como executar

### 1. Clonar o repositório e instalar dependências

```bash
cd mba-semantic-search-rag
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Iniciar o PostgreSQL com pgVector

```bash
docker-compose up -d
```

### 3. Configurar variáveis de ambiente

Criar arquivo `.env` na raiz do projeto:

```
PDF_PATH=document.pdf
COLLECTION_NAME=documents
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag

# Escolha um dos dois:
GOOGLE_API_KEY=sua_chave_aqui  # Para Google Generative AI
# OU
OPENAI_API_KEY=sua_chave_aqui  # Para OpenAI
```

### 4. Ingerir documento PDF

```bash
cd src
python ingest.py
```

Isso carregará o PDF, criará embeddings e armazenará no banco de dados.

### 5. Interagir com o CLI

```bash
python chat.py
```

## Estrutura do projeto

- `src/ingest.py` - Carrega PDF e cria embeddings
- `src/search.py` - Busca semântica de documentos
- `src/chat.py` - Interface CLI
- `src/providers.py` - Resolução dos provedores de IA (`google` ou `open-ai`)
- `src/env.py` - Resolução das variáveis de ambiente
