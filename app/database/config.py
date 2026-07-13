import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smart_doc_assistant.db")

# For SQLite, we need connect_args={"check_same_thread": False} to avoid
# threading issues across different requests/threads in FastAPI.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy ORM models (SQLAlchemy 2.0 style).
    """
    pass

def get_db():
    """
    Dependency generator that yields a database session and ensures
    it is closed after the request is completed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
