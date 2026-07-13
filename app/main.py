from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router

app = FastAPI(
    title="Smart Document Assistant API",
    description="Production-grade agentic QA system for uploaded policies, manuals, and documents.",
    version="1.0.0"
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
