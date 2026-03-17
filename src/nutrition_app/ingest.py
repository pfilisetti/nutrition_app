import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

DOCS_DIR = "docs"
COLLECTION_NAME = "nutrition_docs"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def load_documents(docs_dir):
    documents = []
    if not os.path.exists(docs_dir):
        print(f"Directory '{docs_dir}' not found.")
        return documents

    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        try:
            if filename.endswith(".docx"):
                print(f"Loading {filename}...")
                documents.extend(Docx2txtLoader(filepath).load())
            elif filename.endswith(".xlsx"):
                print(f"Loading {filename}...")
                documents.extend(UnstructuredExcelLoader(filepath).load())
            else:
                print(f"Skipping {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents


def main():
    print("Loading documents...")
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("No documents loaded. Exiting.")
        return
    print(f"Loaded {len(docs)} document(s).")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
