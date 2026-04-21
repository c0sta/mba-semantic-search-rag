import os

from dotenv import load_dotenv

load_dotenv()

PDF_PATH: str = os.getenv("PDF_PATH", "document.pdf")

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
