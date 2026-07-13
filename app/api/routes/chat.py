from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.config import get_db
from app.database.models import ChatMessage as DbChatMessage
from app.api.schemas import ChatRequest, ChatResponse, ChatMessageResponse
from app.services.agent import chat_with_agent

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def run_chat(
    payload: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Accepts user query, invokes the agent execution loop with persistent memory,
    logs user prompts & agent outputs to SQLite, and returns output + reasoning steps.
    Declared using standard 'def' to run on FastAPI's thread pool and prevent blocking the async event loop.
    """
    try:
        result = chat_with_agent(
            session_id=payload.session_id,
            user_message=payload.message,
            provider=payload.provider,
            model_name=payload.model_name,
            db=db
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent orchestration failed: {str(e)}"
        )


@router.get("/{session_id}/history", response_model=List[ChatMessageResponse])
def get_session_history(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieves chronological conversation history for a given session.
    """
    try:
        messages = (
            db.query(DbChatMessage)
            .filter(DbChatMessage.session_id == session_id)
            .order_by(DbChatMessage.timestamp.asc())
            .all()
        )
        return messages
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}"
        )
