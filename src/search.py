import os

from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

DEFAULT_FALLBACK = "Não tenho informações necessárias para responder sua pergunta."

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_EMBEDDING_MODEL: str = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview"
)
GOOGLE_LLM_MODEL: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-flash")
COLLECTION_NAME: str = os.getenv("PG_VECTOR_COLLECTION_NAME", "challenge-collection")
CONNECTION_URL: str = os.getenv("PGVECTOR_URL", "")

for k in ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "GOOGLE_LLM_MODEL"):
    if not os.getenv(k):
        raise ValueError(f"Missing environment variable: {k}")


def _search_vector_db(question: str = ""):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    store = PGVector(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_URL,
        use_jsonb=True,
    )

    return store.similarity_search_with_score(question, k=3)


def _build_context(result) -> str:
    context_parts: list[str] = []

    for doc, _score in result:
        if doc.page_content and doc.page_content.strip():
            context_parts.append(doc.page_content.strip())

    return "\n\n---\n\n".join(context_parts)


def _generate_answer(question: str, context: str) -> str:
    if not context.strip():
        return DEFAULT_FALLBACK

    llm = GoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    prompt = PROMPT_TEMPLATE.format(
        contexto=context,
        pergunta=question,
    )

    response = llm.invoke(prompt)
    answer = str(response).strip()

    if not answer:
        return DEFAULT_FALLBACK

    return answer


def search_prompt(question: str = "") -> str:
    result = _search_vector_db(question)
    context = _build_context(result)
    answer = _generate_answer(question=question, context=context)
    return answer


def _log_search_result(result: str):
    print(f"RESPOSTA: {result}")
