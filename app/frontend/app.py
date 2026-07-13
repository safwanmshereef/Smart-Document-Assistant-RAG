import os
import uuid
import requests
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style Injection for Rich Aesthetics
st.markdown("""
<style>
    /* CSS injections for sleek cards and animations */
    .stApp {
        background-color: #0E1117;
    }
    .document-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #1E232E;
        margin-bottom: 8px;
        border-left: 4px solid #4F8BF9;
    }
    .reasoning-box {
        background-color: #1A1C23;
        border: 1px solid #323544;
        border-radius: 8px;
        padding: 12px;
        font-family: monospace;
        color: #E2E8F0;
        margin-top: 5px;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allowed_html=True)

# Load Backend URL Dynamically from Environment
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar Content
with st.sidebar:
    st.title("🤖 Document Assistant")
    st.write("---")

    # Document uploader
    st.subheader("📁 Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF or text file",
        type=["pdf", "txt"],
        help="Upload corporate policy documents or manuals for the RAG pipeline."
    )

    if uploaded_file is not None:
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner("Processing & indexing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_API_URL}/documents/upload", files=files)
                    if response.status_code == 201:
                        st.success("Document ingested successfully!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.write("---")

    # Registry listing
    st.subheader("📚 Ingested Documents")
    try:
        doc_response = requests.get(f"{BACKEND_API_URL}/documents")
        if doc_response.status_code == 200:
            docs = doc_response.json()
            if docs:
                for doc in docs:
                    st.markdown(f"""
                    <div class="document-card">
                        <span style='font-weight: 600; color: #E2E8F0;'>📄 {doc['filename']}</span><br/>
                        <span style='font-size: 0.8rem; color: #94A3B8;'>Uploaded: {doc['upload_timestamp'][:16].replace('T', ' ')}</span>
                    </div>
                    """, unsafe_allowed_html=True)
            else:
                st.info("No documents uploaded yet.")
        else:
            st.error("Failed to load document registry.")
    except Exception as e:
        st.error(f"Error fetching documents: {e}")

    st.write("---")

    # Reset Session / Clear Chat
    st.subheader("⚙️ Session Controls")
    st.markdown(f"**Session ID:** `{st.session_state['session_id'][:8]}...`")
    if st.button("Clear Chat & New Session", use_container_width=True, type="primary"):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.success("New session generated.")
        st.rerun()

# Main Area Header
st.title("💬 Smart Document Assistant")
st.markdown("Query uploaded company documents, perform calculations, and access live web facts.")

# Display existing chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("reasoning"):
            with st.expander("🔍 View Agent Reasoning"):
                for idx, step in enumerate(msg["reasoning"]):
                    st.markdown(f"**Step {idx+1}:** Called `{step.get('tool')}`")
                    st.json(step.get("tool_input"))
                    st.markdown(f"""<div class="reasoning-box">{step.get('output')}</div>""", unsafe_allowed_html=True)

# Capture user prompt
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user input
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Post user input to API backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "session_id": st.session_state["session_id"],
                    "message": prompt,
                    "provider": "google"
                }
                # Call /chat backend
                chat_res = requests.post(f"{BACKEND_API_URL}/chat", json=payload)
                if chat_res.status_code == 200:
                    data = chat_res.json()
                    output = data.get("output", "")
                    reasoning = data.get("reasoning_trace", [])

                    # Render answer
                    st.write(output)

                    # Save to state
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": output,
                        "reasoning": reasoning
                    })

                    # Render reasoning expander
                    if reasoning:
                        with st.expander("🔍 View Agent Reasoning"):
                            for idx, step in enumerate(reasoning):
                                st.markdown(f"**Step {idx+1}:** Called `{step.get('tool')}`")
                                st.json(step.get("tool_input"))
                                st.markdown(f"""<div class="reasoning-box">{step.get('output')}</div>""", unsafe_allowed_html=True)
                else:
                    err_msg = chat_res.json().get('detail', 'Unknown error')
                    st.error(f"Error from assistant: {err_msg}")
            except Exception as e:
                st.error(f"Connection error: {e}")
