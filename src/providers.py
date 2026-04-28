from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from env import (
    GOOGLE_API_KEY,
    GOOGLE_EMBEDDING_MODEL,
    GOOGLE_LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_LLM_MODEL,
)


def AIProviderResolution():
    if GOOGLE_API_KEY:
        return {
            "provider": "google",
            "embeddings": GoogleGenerativeAIEmbeddings(
                model=GOOGLE_EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
            ),
            "llm": GoogleGenerativeAI(
                model=GOOGLE_LLM_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0,
            ),
        }

    if OPENAI_API_KEY:
        return {
            "provider": "openai",
            "embeddings": OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                api_key=OPENAI_API_KEY,
            ),
            "llm": ChatOpenAI(
                model=OPENAI_LLM_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0,
            ),
        }

    raise ValueError(
        "No AI provider API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY."
    )
