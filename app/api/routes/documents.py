import os
import tempfile
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.database.config import get_db
from app.database.models import Document as DbDocument
from app.api.schemas import DocumentResponse
from app.services.document_processor import ingest_document
from app.services.vector_store import ingest_chunks, delete_document_vectors

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a document (.pdf or .txt), spools it safely to disk,
    processes & chunks it, registers it in the SQLite metadata DB,
    indexes it in ChromaDB, and performs a guaranteed file cleanup on complete.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only .pdf and .txt files are supported."
        )

    temp_path = None
    try:
        # Create a temporary file and write the uploaded bytes to it
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = file.file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process and ingest using the injected DB session
        db_id, chunks = ingest_document(temp_path, filename, db=db)
        
        # Embed chunks in ChromaDB
        ingest_chunks(chunks)

        return {
            "document_id": db_id,
            "filename": filename,
            "status": "success",
            "message": "Document uploaded and indexed successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and upload document: {str(e)}"
        )
    finally:
        # Guarantee cleanup of temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.get("", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """
    Retrieves all ingested documents and their metadata from SQLite.
    """
    try:
        documents = db.query(DbDocument).order_by(DbDocument.upload_timestamp.desc()).all()
        return documents
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Deletes an ingested document from both SQLite database and ChromaDB vector store.
    """
    doc = db.query(DbDocument).filter(DbDocument.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )
    try:
        # Delete from ChromaDB
        delete_document_vectors(doc.filename)
        # Delete from SQLite
        db.delete(doc)
        db.commit()
        return {
            "status": "success",
            "message": f"Document '{doc.filename}' deleted successfully."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
