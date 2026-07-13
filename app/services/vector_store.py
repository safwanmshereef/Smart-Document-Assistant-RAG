import os
from typing import List

# Import base LangChain classes
from langchain_core.documents import Document

# Fallback for HuggingFaceEmbeddings to ensure version compatibility
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Fallback for ChromaDB VectorStore to ensure version compatibility
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Base configurations
PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
COLLECTION_NAME = "smart_docs"

# Resolve the local cached path to sentence-transformers/all-MiniLM-L6-v2 to run offline
LOCAL_MODEL_PATH = "C:\\Users\\safwa\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2\\snapshots\\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

model_name = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "all-MiniLM-L6-v2"

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name=model_name)

def get_vector_store() -> Chroma:
    """
    Initializes and returns the persistent Chroma vector store instance.
    Uses the directory app/chroma_db and collection smart_docs.

    Returns:
        An initialized Chroma vector store instance.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

def ingest_chunks(chunks: List[Document]) -> List[str]:
    """
    Ingests chunked Document objects into ChromaDB.
    Attaches metadata such as source file and page/chunk ID for citation.

    Args:
        chunks: List of LangChain Document objects representing text chunks.

    Returns:
        List of generated chunk IDs in ChromaDB.
    """
    if not chunks:
        return []

    vector_store = get_vector_store()

    # Ensure metadata fields are fully populated to support citations
    for idx, chunk in enumerate(chunks):
        chunk.metadata.setdefault("source", "unknown")
        chunk.metadata.setdefault("page", 1)
        chunk.metadata.setdefault("chunk_index", idx)

    # Add the documents to persistent storage
    ids = vector_store.add_documents(chunks)
    return ids

def retrieve_documents(query: str, top_k: int = 4) -> List[Document]:
    """
    Performs a similarity search in ChromaDB and returns relevant chunks
    alongside their citation metadata.

    Args:
        query: The search query string.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        A list of retrieved LangChain Document objects.
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=top_k)

def clear_vector_store() -> None:
    """
    Deletes the current collection in ChromaDB to clean up and reset state.
    """
    vector_store = get_vector_store()
    try:
        vector_store.delete_collection()
    except Exception:
        # Ignore errors if the collection does not exist
        pass
