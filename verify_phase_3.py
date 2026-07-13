import os
import sys
import logging
import uuid
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
logger = logging.getLogger("Phase3Verifier")

def log_success(msg: str):
    logger.info(f"{GREEN}[SUCCESS] {msg}{RESET}")

def log_error(msg: str):
    logger.error(f"{RED}[ERROR] {msg}{RESET}")

def log_warning(msg: str):
    logger.warning(f"{YELLOW}[WARNING] {msg}{RESET}")

def log_info(msg: str):
    logger.info(f"{BLUE}[INFO] {msg}{RESET}")

def clean_sqlite_session_data(session_id: str, doc_id: str = None):
    try:
        from app.database.config import SessionLocal
        from app.database.models import Document, Session, ChatMessage
        
        db = SessionLocal()
        try:
            # Delete messages
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            # Delete session
            db.query(Session).filter(Session.id == session_id).delete()
            # Delete doc if present
            if doc_id:
                db.query(Document).filter(Document.id == doc_id).delete()
            db.commit()
            log_info(f"SQLite records cleaned up for session: {session_id}")
        finally:
            db.close()
    except Exception as e:
        log_warning(f"Error during SQLite database cleanup: {e}")

def run_verification():
    log_info("=== SMART DOCUMENT ASSISTANT: PHASE 3 VERIFICATION RUN ===")

    load_dotenv()
    
    # Verify GOOGLE_API_KEY
    if not os.getenv("GOOGLE_API_KEY"):
        log_error("GOOGLE_API_KEY is not configured in .env. Failing verification suite.")
        sys.exit(1)

    # 1. Initialize DB and Session UUID
    session_id = str(uuid.uuid4())
    log_info(f"Generated test Session UUID: {session_id}")
    
    # Create SQLite Session record
    try:
        from app.database.config import SessionLocal, Base, engine
        from app.database.models import Session as DbSession
        
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        
        db = SessionLocal()
        try:
            db_session = DbSession(id=session_id)
            db.add(db_session)
            db.commit()
            log_success("SQLite test session initialized.")
        finally:
            db.close()
    except Exception as e:
        log_error(f"Failed to initialize SQLite session: {e}")
        sys.exit(1)

    # 2. Ingest budget document into ChromaDB
    filename = "fictional_budget_guidelines.txt"
    file_path = os.path.abspath(filename)
    budget_content = "Acme Corp Financial Update: The corporate Q3 marketing budget has been officially finalized at $50,000."
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(budget_content)
            
        from app.services.document_processor import ingest_document
        from app.services.vector_store import ingest_chunks
        
        doc_id, chunks = ingest_document(file_path, filename)
        log_success(f"Budget document ingested to SQLite. Document ID: {doc_id}")
        
        chunk_ids = ingest_chunks(chunks)
        log_success(f"Ingested {len(chunk_ids)} budget chunks into ChromaDB.")
        
    except Exception as e:
        log_error(f"Document ingestion failed: {e}")
        clean_sqlite_session_data(session_id)
        if os.path.exists(file_path):
            os.remove(file_path)
        sys.exit(1)

    # 3. Fire multi-step tool-use query
    try:
        from app.services.agent import chat_with_agent
        
        query = "What is 15% of the corporate Q3 marketing budget mentioned in the document guidelines?"
        log_info(f"Firing query to chat_with_agent: '{query}'")
        
        # Run using Google Gemini model
        res = chat_with_agent(session_id=session_id, user_message=query, provider="google")
        
        output = res.get("output", "")
        reasoning_trace = res.get("reasoning_trace", [])
        
        log_info(f"Agent Final Output: {output}")
        log_info(f"Agent Reasoning Trace: {reasoning_trace}")
        
        # Assertions for tool execution
        has_search = any(step["tool"] == "search_documents" for step in reasoning_trace)
        has_calc = any(step["tool"] == "calculator" for step in reasoning_trace)
        
        assert has_search, "Agent failed to execute search_documents tool."
        assert has_calc, "Agent failed to execute calculator tool."
        log_success("Reasoning trace asserts that search_documents and calculator were executed.")
        
        # Assertions for calculation correctness (yielding 7500)
        assert "7500" in output or "7,500" in output, f"Agent response did not yield the correct budget calculation result of $7,500. Output was: {output}"
        log_success("Math calculation output yields exactly $7,500.")
        
        # Assertions for citations in the final answer
        assert "Source" in output or "Page" in output or "fictional_budget_guidelines.txt" in output, "Agent output is missing source citations."
        log_success("Citation presence verified in the agent's response.")
        
    except AssertionError as ae:
        log_error(f"Integrations / Math / Citation Assertions Failed: {ae}")
        cleanup(session_id, doc_id, file_path)
        sys.exit(1)
    except Exception as e:
        log_error(f"Agent conversation step 1 failed: {e}")
        cleanup(session_id, doc_id, file_path)
        sys.exit(1)

    # 4. Refusal Check (Anti-Hallucination Guardrail)
    try:
        refusal_query = "What is the CEO's favorite color?"
        log_info(f"Firing out-of-context query to verify anti-hallucination: '{refusal_query}'")
        
        refusal_res = chat_with_agent(session_id=session_id, user_message=refusal_query, provider="google")
        refusal_output = refusal_res.get("output", "")
        log_info(f"Agent Refusal Output: {refusal_output}")
        
        # Assert that output contains refusal language
        assert any(phrase in refusal_output.lower() for phrase in ["i don't know", "do not contain", "not contain", "don't know", "cannot find", "no information"]), \
            f"Agent did not refuse query. Output was: {refusal_output}"
        log_success("Anti-hallucination check passed: Agent correctly refused to answer out-of-context query.")
        
    except AssertionError as ae:
        log_error(f"Anti-Hallucination Guardrail Assertion Failed: {ae}")
        cleanup(session_id, doc_id, file_path)
        sys.exit(1)
    except Exception as e:
        log_error(f"Agent refusal query failed: {e}")
        cleanup(session_id, doc_id, file_path)
        sys.exit(1)

    # 5. Dispatch follow-up memory and custom tool query: "What day is it today?"
    try:
        followup_query = "What day is it today?"
        log_info(f"Firing follow-up datetime query: '{followup_query}'")
        
        datetime_res = chat_with_agent(session_id=session_id, user_message=followup_query, provider="google")
        datetime_output = datetime_res.get("output", "")
        datetime_trace = datetime_res.get("reasoning_trace", [])
        
        log_info(f"Agent Datetime Output: {datetime_output}")
        log_info(f"Agent Datetime Reasoning Trace: {datetime_trace}")
        
        # Check that it used get_current_date_time
        has_dt_tool = any(step["tool"] == "get_current_date_time" for step in datetime_trace)
        assert has_dt_tool, "Agent failed to execute get_current_date_time tool."
        
        # Check database logs to verify memory retention in session
        from app.database.models import ChatMessage as DbChatMessage
        db = SessionLocal()
        try:
            stored_msgs = db.query(DbChatMessage).filter(DbChatMessage.session_id == session_id).all()
            log_info(f"Total stored chat messages in SQLite for session {session_id}: {len(stored_msgs)}")
            # Expecting 6 messages: user-query1, agent-response1, user-query2, agent-response2, user-query3, agent-response3
            assert len(stored_msgs) == 6, f"SQLite memory logs missing messages. Count: {len(stored_msgs)}"
            log_success("SQLite persistent conversation memory verified (all messages logged).")
        finally:
            db.close()
            
    except AssertionError as ae:
        log_error(f"Memory / Datetime Assertions Failed: {ae}")
        cleanup(session_id, doc_id, file_path)
        sys.exit(1)
    except Exception as e:
        log_error(f"Agent datetime/memory verification step failed: {e}")
        cleanup(session_id, doc_id, file_path)
        sys.exit(1)

    # 6. Cleanup
    cleanup(session_id, doc_id, file_path)
    log_success("All Phase 3 verifications completed successfully!")
    sys.exit(0)

def cleanup(session_id: str, doc_id: str, file_path: str):
    log_info("Cleaning up verification test artifacts...")
    # Delete temporary text file
    if os.path.exists(file_path):
        os.remove(file_path)
        log_info("Temporary budget file removed.")
        
    # Delete SQLite records
    clean_sqlite_session_data(session_id, doc_id)
    
    # Clear ChromaDB vector collection
    try:
        from app.services.vector_store import clear_vector_store
        clear_vector_store()
        log_info("ChromaDB vector store cleared.")
    except Exception as e:
        log_warning(f"Failed to clear ChromaDB vector store: {e}")

if __name__ == "__main__":
    run_verification()
