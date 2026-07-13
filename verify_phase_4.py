import os
import sys
import logging
import uuid
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# ANSI escape codes for coloring output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Phase4Verifier")

def log_success(msg: str):
    logger.info(f"{GREEN}[SUCCESS] {msg}{RESET}")

def log_error(msg: str):
    logger.error(f"{RED}[ERROR] {msg}{RESET}")

def log_warning(msg: str):
    logger.warning(f"{YELLOW}[WARNING] {msg}{RESET}")

def log_info(msg: str):
    logger.info(f"{BLUE}[INFO] {msg}{RESET}")

def run_verification():
    log_info("=== SMART DOCUMENT ASSISTANT: PHASE 4 VERIFICATION RUN ===")

    load_dotenv()
    
    # 1. Import app
    try:
        from app.main import app
        client = TestClient(app)
        log_success("FastAPI TestClient initialized successfully.")
    except Exception as e:
        log_error(f"Failed to import app and initialize TestClient: {e}")
        sys.exit(1)

    # Verify Healthcheck Endpoint
    try:
        res = client.get("/")
        assert res.status_code == 200
        assert res.json().get("status") == "ok"
        log_success("Root health check verified successfully.")
    except Exception as e:
        log_error(f"Healthcheck endpoint failed: {e}")
        sys.exit(1)

    # 2. Create a dummy .txt policy file
    filename = "fictional_q3_budget_guidelines.txt"
    file_path = os.path.abspath(filename)
    budget_content = "Acme Corp Financial Update: The corporate Q3 marketing budget has been officially finalized at $50,000."
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(budget_content)
        log_info(f"Created temporary dummy policy file: {filename}")
    except Exception as e:
        log_error(f"Failed to create dummy file: {e}")
        sys.exit(1)

    doc_id = None
    session_id = str(uuid.uuid4())

    try:
        # 3. Test POST /documents/upload
        log_info("Testing POST /documents/upload...")
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "text/plain")}
            upload_res = client.post("/documents/upload", files=files)
            
        assert upload_res.status_code == 201, f"Expected 201 Created, got {upload_res.status_code}. Response: {upload_res.text}"
        upload_data = upload_res.json()
        doc_id = upload_data.get("document_id")
        assert doc_id, "Response does not contain a document ID."
        log_success(f"Document uploaded and indexed successfully. Doc ID: {doc_id}")

        # 4. Test GET /documents
        log_info("Testing GET /documents registry...")
        list_res = client.get("/documents")
        assert list_res.status_code == 200, f"Expected 200 OK, got {list_res.status_code}"
        list_data = list_res.json()
        filenames = [doc.get("filename") for doc in list_data]
        assert filename in filenames, f"Uploaded document {filename} not listed in the document registry: {filenames}"
        log_success("Document registry lists the uploaded file.")

        # 5. Test POST /chat - RAG Query
        log_info(f"Testing POST /chat with Session ID: {session_id}...")
        chat_payload = {
            "session_id": session_id,
            "message": "What is 15% of the corporate Q3 marketing budget mentioned in the document guidelines?",
            "provider": "google",
            "model_name": "gemini-3.1-flash-lite"
        }
        chat_res = client.post("/chat", json=chat_payload)
        assert chat_res.status_code == 200, f"Expected 200 OK, got {chat_res.status_code}. Response: {chat_res.text}"
        
        chat_data = chat_res.json()
        output = chat_data.get("output", "")
        reasoning = chat_data.get("reasoning_trace", [])
        
        log_info(f"Agent Chat Output: {output}")
        log_info(f"Agent Reasoning Trace: {reasoning}")
        
        assert "7500" in output or "7,500" in output, f"Calculation failed. Final output: {output}"
        # Assertions for tool execution
        has_search = any(step["tool"] == "search_documents" for step in reasoning)
        has_calc = any(step["tool"] == "calculator" for step in reasoning)
        assert has_search, "Agent failed to execute search_documents tool."
        assert has_calc, "Agent failed to execute calculator tool."
        log_success("Multi-step RAG chat calculation and reasoning trace verified.")

        # Cool down to prevent rate limits
        import time
        log_info("Sleeping 5 seconds for rate limit cooldown...")
        time.sleep(5)

        # 6. Test POST /chat - Web Search Query
        web_query = "What is the current stock price of Apple?"
        log_info(f"Testing POST /chat with web search query: '{web_query}'...")
        web_payload = {
            "session_id": session_id,
            "message": web_query,
            "provider": "google",
            "model_name": "gemini-3.1-flash-lite"
        }
        web_res = client.post("/chat", json=web_payload)
        assert web_res.status_code == 200, f"Expected 200 OK, got {web_res.status_code}. Response: {web_res.text}"
        
        web_data = web_res.json()
        web_output = web_data.get("output", "")
        web_reasoning = web_data.get("reasoning_trace", [])
        
        log_info(f"Web Search Output: {web_output}")
        log_info(f"Web Search Reasoning Trace: {web_reasoning}")
        
        # Verify that web_search tool was invoked
        has_web_search = any(step["tool"] == "web_search" for step in web_reasoning)
        assert has_web_search, "Agent failed to execute web_search tool."
        log_success("Web search query executed and web_search tool invocation verified.")

        # 7. Test GET /chat/{session_id}/history
        log_info("Testing GET /chat/{session_id}/history...")
        history_res = client.get(f"/chat/{session_id}/history")
        assert history_res.status_code == 200, f"Expected 200 OK, got {history_res.status_code}"
        history_data = history_res.json()
        log_info(f"Retrieved History: {history_data}")
        # Expected 4 messages: user-query1, agent-response1, user-query2, agent-response2
        assert len(history_data) == 4, f"Expected 4 history messages, got {len(history_data)}"
        log_success("Chronological conversation history successfully retrieved and validated.")

    except AssertionError as ae:
        log_error(f"API Integration Assertion Failed: {ae}")
        cleanup(session_id, doc_id, file_path)
        sys.exit(1)
    except Exception as e:
        log_error(f"API Integration Test Failed with Exception: {e}")
        cleanup(session_id, doc_id, file_path)
        sys.exit(1)

    cleanup(session_id, doc_id, file_path)
    log_success("All Phase 4 API and tool verification checks passed successfully!")
    sys.exit(0)

def cleanup(session_id: str, doc_id: str, file_path: str):
    log_info("Cleaning up verification test artifacts...")
    # Delete temporary text file
    if os.path.exists(file_path):
        os.remove(file_path)
        log_info("Temporary budget guidelines file removed.")
        
    # Delete SQLite records
    try:
        from app.database.config import SessionLocal
        from app.database.models import Document, Session, ChatMessage
        db = SessionLocal()
        try:
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            db.query(Session).filter(Session.id == session_id).delete()
            if doc_id:
                db.query(Document).filter(Document.id == doc_id).delete()
            db.commit()
            log_info("SQLite records cleaned up successfully.")
        finally:
            db.close()
    except Exception as e:
        log_warning(f"Error cleaning SQLite records: {e}")
        
    # Clear ChromaDB vector collection
    try:
        from app.services.vector_store import clear_vector_store
        clear_vector_store()
        log_info("ChromaDB vector store cleared.")
    except Exception as e:
        log_warning(f"Failed to clear ChromaDB: {e}")

if __name__ == "__main__":
    run_verification()
