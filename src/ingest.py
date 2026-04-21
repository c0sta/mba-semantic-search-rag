import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH: str = os.getenv("PDF_PATH", "document.pdf")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_EMBEDDING_MODEL: str = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview"
)
COLLECTION_NAME: str = os.getenv("PG_VECTOR_COLLECTION_NAME", "challenge-collection")
CONNECTION_URL: str = os.getenv("PGVECTOR_URL", "")

for k in ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "GOOGLE_LLM_MODEL"):
    if not os.getenv(k):
        raise ValueError(f"Missing environment variable: {k}")


def ingest_pdf():
    docs = PyPDFLoader(PDF_PATH).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=False
    ).split_documents(docs)

    if not splits:
        raise SystemExit("No splits generated from the document.")

    enriched = [
        Document(
            page_content=split.page_content,
            metadata={k: v for k, v in split.metadata.items() if v not in ("", None)},
        )
        for split in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

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

    store.add_documents(enriched, ids=ids)
    print("Documents added to the vector store successfully.")


if __name__ == "__main__":
    ingest_pdf()
