import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

DOCS_DIR = "docs_md"
COLLECTION_NAME = "nutrition_docs"
EMBEDDING_DIM = 1024  # intfloat/multilingual-e5-large

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def load_documents(docs_dir):
    documents = []
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Directory '{docs_dir}' not found.")
        return documents

    for md_file in sorted(docs_path.rglob("*.md")):
        try:
            print(f"Loading {md_file}...")
            documents.extend(TextLoader(str(md_file), encoding="utf-8").load())
        except Exception as e:
            print(f"Error loading {md_file}: {e}")

    return documents


def split_documents(documents):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata["source"] = doc.metadata.get("source", "")
        chunks.extend(splits)
    return chunks


def main():
    print("Loading documents...")
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("No documents loaded. Exiting.")
        return
    print(f"Loaded {len(docs)} file(s).")

    print("Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embedding model (intfloat/multilingual-e5-large)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — deleting and re-creating.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    print(f"Storing {len(chunks)} chunks in Qdrant collection '{COLLECTION_NAME}'...")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        collection_name=COLLECTION_NAME,
    )

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
