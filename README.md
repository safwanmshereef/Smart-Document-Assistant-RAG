# Smart Document Assistant RAG Pipeline

A production-grade, containerized Retrieval-Augmented Generation (RAG) agentic QA assistant. This project combines a **FastAPI** backend, **Streamlit** frontend, **LangChain** agent orchestration, **ChromaDB** vector storage, and **SQLite** persistent chat history.

---

## 🛠️ Architecture Overview

The application follows a modular, PEP 8 compliant, service-oriented architecture:

```mermaid
graph TD
    A[Streamlit Frontend] -->|HTTP Request| B[FastAPI Web API]
    B -->|Router Injection| C[API Routes /documents & /chat]
    C -->|relational operations| D[SQLite via SQLAlchemy]
    C -->|agentic loop orchestration| E[LangChain Agent Executor]
    E -->|Search Tool| F[Vector Store: ChromaDB]
    E -->|Math Tool| G[AST-Based Secure Calculator]
    E -->|Live Web Tool| H[DuckDuckGo Search]
    E -->|Time Tool| I[System Date-Time]
```

- **Frontend client**: Built with Streamlit, exposing document uploading, metadata registries, real-time message feeds, and expanders showing structural reasoning traces.
- **API backend**: FastAPI web service using dependency injection (`Depends(get_db)`) to manage SQLite connections and standard `def` execution patterns to prevent blocking the event loop.
- **Agent core**: ReAct-style LangChain agent leveraging Gemini (`gemini-3.5-flash` or `gemini-3.1-flash-lite`) and local Ollama model options (`llama3.2:3b`).
- **Memory storage**: Persistent SQLite database storing Uploaded Document registries, Sessions, and chronological ChatMessage history.
- **Semantic retrieval**: ChromaDB persistent vector database utilizing `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to run similarity matches.

---

## ⚙️ Design Decisions

### 1. Chunking Strategy (RecursiveCharacterTextSplitter)
- **Choice**: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
- **Justification**: Using character-recursive splitting ensures semantic continuity across paragraph boundaries. A `200` character overlap guarantees context isn't lost at boundaries, preserving the semantic coherence required by the embedding model.

### 2. Embedding Model (all-MiniLM-L6-v2)
- **Choice**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional).
- **Justification**: Offers a balance of execution speed and quality. By resolving local cache directories (at `~/.cache/huggingface`), it runs fully local, bypassing network dependencies and protecting local VRAM/RAM constraints.

### 3. Agent Framework (ReAct Tool-Calling)
- **Choice**: LangChain Tool-Calling Agent (`create_tool_calling_agent`).
- **Justification**: Standard ReAct architectures allow the model to reason about what tools to invoke. By returning intermediate steps, we can expose the complete reasoning trace to users. Docstrings are prompt-engineered to enforce input formatting (e.g. preventing calculator text-injection vulnerabilities).

---

## 🚀 Setup & Execution

### Option A: Running via Docker (Recommended - 1-Click Launch)

Ensure you have [Docker](https://www.docker.com/) and Docker Compose installed.

1. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

2. **Launch Services**:
   Run the following command:
   ```bash
   docker-compose up --build
   ```

3. **Access Applications**:
   - **Streamlit Frontend**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Running Locally (Virtual Environment)

1. **Set Up Python Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows Activation
   .\.venv\Scripts\activate
   # Linux/MacOS Activation
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. **Run FastAPI Backend**:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

5. **Run Streamlit Frontend** (in a new terminal):
   ```bash
   streamlit run app/frontend/app.py --server.port 8501
   ```

---

## 🧪 Verification Runs

Verification suites are included in the root directory:
- Run **FastAPI Backend verification**: `python verify_phase_4.py`
- Run **Stateful Agent loop verification**: `python verify_phase_3.py`

---

## 💬 Sample Queries to Test

1. **Multi-Tool Budget RAG Calculation**:
   > "What is 15% of the corporate Q3 marketing budget mentioned in the document guidelines?"
   *(Triggers `search_documents` followed by `calculator`)*

2. **Out-of-Context Refusal (Anti-Hallucination)**:
   > "What is the CEO's favorite color?"
   *(Triggers `search_documents` and returns "I don't know" refusal language)*

3. **Live Web Search**:
   > "What is the current stock price of Apple (AAPL)?"
   *(Triggers `web_search` to query live web data)*

4. **Document Topic Summarization**:
   > "Provide a broad summary of the corporate marketing strategies mentioned in the policies."
   *(Triggers `summarize_document_topic` using a larger top_k chunk retrieval window)*

5. **Local Datetime Reference**:
   > "What is the current date and time?"
   *(Triggers `get_current_date_time`)*

---

## ⚠️ Known Limitations & Future Roadmap

If provided with more development time, the following improvements would be prioritized:
1. **Vector Scale**: Transition ChromaDB to an enterprise pgvector PostgreSQL service for distributed querying.
2. **Caching Layer**: Integrate Redis to cache common semantic queries and avoid redundant LLM invocations.
3. **Async Streaming**: Implement FastAPI WebSockets or Server-Sent Events (SSE) to stream the agent's textual response character-by-character.
4. **Enhanced Security**: Deploy an API gateway with rate-limiting, authentication tokens, and payload filtering.
