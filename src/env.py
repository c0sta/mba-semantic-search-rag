import os

from dotenv import load_dotenv

load_dotenv()

PDF_PATH: str = os.getenv("PDF_PATH", "document.pdf")

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_EMBEDDING_MODEL: str = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview"
)
GOOGLE_LLM_MODEL: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL: str = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

COLLECTION_NAME: str = os.getenv("PG_VECTOR_COLLECTION_NAME", "challenge-collection")
CONNECTION_URL: str = os.getenv("PGVECTOR_URL", "")

for k in ("PGVECTOR_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise ValueError(f"Missing environment variable: {k}")
