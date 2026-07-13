import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.database.config import SessionLocal
from app.database.models import Document as DbDocument

def save_document_metadata(filename: str) -> str:
    """
    Saves document metadata to SQLite database and returns the generated UUID.

    Args:
        filename: The original file name.

    Returns:
        The generated UUID string of the Document record in the database.
    """
    db = SessionLocal()
    try:
        db_doc = DbDocument(filename=filename)
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        return db_doc.id
    finally:
        db.close()

def process_document(file_path: str, filename: str) -> List[Document]:
    """
    Loads and extracts text from a file. Supports .pdf and .txt files.
    Performs fast-fail validation on file extensions before invoking loaders.

    Args:
        file_path: Path to the target file.
        filename: Original filename to store in citation metadata.

    Returns:
        A list of LangChain Document objects.

    Raises:
        ValueError: If file is not a PDF or TXT.
        FileNotFoundError: If the file does not exist at file_path.
    """
    # Fast-fail extension check
    _, ext = os.path.splitext(filename.lower())
    if ext not in (".pdf", ".txt"):
        raise ValueError(
            f"Unsupported file format: '{ext}'. Only .pdf and .txt files are supported."
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    # Use the appropriate LangChain document loader
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()

    # Normalize metadata to ensure citation info is present
    for doc in docs:
        doc.metadata["source"] = filename
        if ext == ".pdf":
            # Convert 0-indexed page to 1-indexed for human-readable citation
            page_val = doc.metadata.get("page", 0) + 1
            doc.metadata["page"] = page_val
        else:
            # Default TXT files to page 1
            doc.metadata["page"] = 1

    return docs

def chunk_document_text(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Chunks text using a RecursiveCharacterTextSplitter.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Target overlap size in characters.

    Returns:
        A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    
    # Annotate chunks with chunk IDs for citation
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        
    return chunks

def ingest_document(
    file_path: str, 
    filename: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> Tuple[str, List[Document]]:
    """
    Orchestrates the full document ingestion process:
    1. Validates and loads the document pages.
    2. Chunks the document pages.
    3. Saves document metadata to the SQLite database.
    4. Links the generated SQLite database document ID to the chunked LangChain documents' metadata.

    Args:
        file_path: Local path to the file.
        filename: Original file name.
        chunk_size: Document split chunk size.
        chunk_overlap: Document split chunk overlap.

    Returns:
        A tuple of (sqlite_document_id, chunked_documents).
    """
    # 1. Load and validate
    docs = process_document(file_path, filename)
    
    # 2. Chunk text
    chunks = chunk_document_text(docs, chunk_size, chunk_overlap)
    
    # 3. Save metadata to relational database
    db_id = save_document_metadata(filename)
    
    # 4. Attach relational database ID to the chunks' metadata as a citation attribute
    for chunk in chunks:
        chunk.metadata["document_id"] = db_id
        
    return db_id, chunks
