from datetime import datetime
import uuid
from sqlalchemy import String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.config import Base

class Document(Base):
    """
    Model representing uploaded documents.
    Stores metadata such as filename and timestamp.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4()),
        index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    upload_timestamp: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename})>"


class Session(Base):
    """
    Model representing a user's chat session.
    A single session can contain multiple chat messages.
    """
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4()),
        index=True
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False
    )

    # One-to-many relationship: deleting a session will cascade delete its messages
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", 
        back_populates="session", 
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, created_at={self.created_at})>"


class ChatMessage(Base):
    """
    Model representing a message in a chat conversation.
    Linked to a parent chat session.
    """
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True, 
        autoincrement=True
    )
    session_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("sessions.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    role: Mapped[str] = mapped_column(
        String(50), 
        nullable=False
    )  # Should represent "user" or "agent"
    content: Mapped[str] = mapped_column(
        Text, 
        nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False
    )

    # Many-to-one relationship back to Session
    session: Mapped["Session"] = relationship(
        "Session", 
        back_populates="messages"
    )

    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, role={self.role})>"
