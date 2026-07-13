from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.config import engine, Base
from app.database.models import Document, Session, ChatMessage  # noqa: F401
from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router


# Create tables on import to support TestClient/verification runs without async lifespan context
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup if they don't exist."""
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Smart Document Assistant API",
    description="Production-grade agentic QA system for uploaded policies, manuals, and documents.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, customize for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root health check endpoint
@app.get("/")
def health_check():
    """
    Root endpoint serving health-check status.
    """
    return {
        "status": "ok",
        "message": "Smart Document Assistant API is healthy"
    }

# Include API routers
app.include_router(documents_router)
app.include_router(chat_router)
