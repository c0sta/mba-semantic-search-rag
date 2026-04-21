from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env import COLLECTION_NAME, CONNECTION_URL, PDF_PATH
from providers import AIProviderResolution


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

    provider = AIProviderResolution()
    embeddings = provider["embeddings"]

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
