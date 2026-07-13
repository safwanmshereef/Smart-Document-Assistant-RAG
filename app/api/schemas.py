from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

class DocumentResponse(BaseModel):
    """
    Pydantic schema representing document metadata response.
    """
    id: str
    filename: str
    upload_timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    """
    Pydantic schema for chat requests.
    """
    session_id: str
    message: str
    provider: str
    model_name: Optional[str] = None
    selected_doc_name: Optional[str] = None
    selected_doc_names: Optional[List[str]] = None


class ChatResponse(BaseModel):
    """
    Pydantic schema for chat response, exposing the reasoning trace.
    """
    output: str
    reasoning_trace: List[Dict[str, Any]]


class ChatMessageResponse(BaseModel):
    """
    Pydantic schema representing individual messages in chat history.
    """
    id: int
    session_id: str
    role: str
    content: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
