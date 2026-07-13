"""
Smart Document Assistant – Production Streamlit Frontend
=========================================================
"""

import os
import uuid
import math
from datetime import datetime

import requests
import requests.exceptions
import streamlit as st
from datetime import datetime, timedelta

def utc_to_ist(utc_str: str) -> str:
    """
    Converts a UTC timestamp string (ISO format) to IST (UTC +5:30) and formats as YYYY-MM-DD HH:MM.
    """
    if not utc_str:
        return ""
    try:
        # Strip decimal microseconds and Z if present
        clean_str = utc_str.split(".")[0].replace("Z", "").replace("T", " ")
        dt = datetime.strptime(clean_str[:19], "%Y-%m-%d %H:%M:%S")
        ist_dt = dt + timedelta(hours=5, minutes=30)
        return ist_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return utc_str[:16].replace("T", " ")

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stDeployButton"] { display: none !important; }
    /* Always show the sidebar collapse/expand toggle */
    [data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #0f1a2e 100%);
        border-right: 1px solid #1e293b;
    }

    .section-header {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #4F8BF9;
        margin: 12px 0 8px 0;
    }

    .doc-card {
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(79, 139, 249, 0.06);
        border-left: 3px solid #4F8BF9;
        margin-bottom: 6px;
        transition: all 0.15s ease;
    }
    .doc-card.selected {
        border-left-color: #34D399;
        background: rgba(52, 211, 153, 0.08);
    }
    .doc-name { font-weight: 600; color: #E2E8F0; font-size: 0.85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .doc-meta { font-size: 0.7rem; color: #64748B; margin-top: 1px; }

    .session-card {
        padding: 8px 10px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid #1e293b;
        margin-bottom: 6px;
        transition: all 0.15s ease;
    }
    .session-card:hover {
        background: rgba(255, 255, 255, 0.06);
    }
    .session-card.active {
        border-color: #4F8BF9;
        background: rgba(79, 139, 249, 0.05);
    }
    .session-title { font-size: 0.8rem; color: #CBD5E1; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .session-meta { font-size: 0.68rem; color: #475569; margin-top: 1px; }

    .reasoning-step {
        background: #0d1526;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #CBD5E1;
        line-height: 1.6;
    }
    .tool-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: monospace;
    }
    .tb-search { background: rgba(99,102,241,0.2); color: #818CF8; }
    .tb-calc { background: rgba(245,158,11,0.2); color: #F59E0B; }
    .tb-web { background: rgba(52,211,153,0.2); color: #34D399; }
    .tb-summary { background: rgba(244,63,94,0.2); color: #F43F5E; }

    .metric-card {
        background: linear-gradient(135deg, #1a2744 0%, #0F172A 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #60A5FA; }
    .metric-label { font-size: 0.75rem; color: #94A3B8; margin-top: 2px; }

    .token-bar-bg {
        background: #1E293B;
        border-radius: 6px;
        height: 6px;
        width: 100%;
        margin-top: 4px;
        overflow: hidden;
    }
    .token-bar-fill {
        height: 6px;
        border-radius: 6px;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        transition: width 0.4s ease;
    }
    .token-bar-warn { background: linear-gradient(90deg, #F59E0B, #EF4444); }

    .provider-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .pb-google { background: rgba(66,133,244,0.15); color: #4285F4; }
    .pb-ollama { background: rgba(52,211,153,0.15); color: #34D399; }

    .tool-info-row {
        display: flex;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .tool-info-name { font-size: 0.78rem; font-weight: 600; color: #CBD5E1; min-width: 110px; }
    .tool-info-desc { font-size: 0.72rem; color: #64748B; }

    /* Floating Chat Input Toolbar Styles */
    .toolbar-anchor { display: none; }
    
    /* Target ONLY the horizontal block (columns) that is the sibling of our toolbar anchor */
    div[data-testid="element-container"]:has(.toolbar-anchor) ~ div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] {
        position: fixed !important;
        bottom: 34px !important; /* Positioned inside the input text field bottom bar */
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 100% !important;
        max-width: 730px !important;
        display: flex !important;
        justify-content: space-between !important; /* Space between model badge on left and tools on right */
        align-items: center !important;
        padding-left: 20px !important;
        padding-right: 52px !important; /* Sit right to the left of the send button inside the input box */
        z-index: 99999 !important;
        pointer-events: none !important;
        background: transparent !important;
    }
    
    /* Target columns inside the floating toolbar */
    div[data-testid="element-container"]:has(.toolbar-anchor) ~ div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] [data-testid="column"] {
        width: auto !important;
        flex: none !important;
        pointer-events: auto !important;
    }

    /* Make the right-hand column display buttons inline-row */
    div[data-testid="element-container"]:has(.toolbar-anchor) ~ div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] [data-testid="column"]:last-child [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 6px !important;
    }
    
    div[data-testid="element-container"]:has(.toolbar-anchor) ~ div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] button {
        pointer-events: auto !important;
        background-color: #1E293B !important;
        color: #CBD5E1 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 0px 8px !important;
        font-size: 0.7rem !important;
        height: 24px !important; /* Compact height to fit inside the textarea */
        min-height: 24px !important;
        line-height: 22px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    }
    
    div[data-testid="element-container"]:has(.toolbar-anchor) ~ div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #334155 !important;
        border-color: #4F8BF9 !important;
        color: #FFF !important;
    }

    /* Model display badge inside the chat input box bottom-left */
    .model-badge-chat {
        background-color: #1E293B !important;
        color: #CBD5E1 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 4px !important;
        height: 24px !important;
        white-space: nowrap !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        pointer-events: auto !important;
    }

    /* ── Premium Custom Chat Input Card Styling ── */
    div[data-testid="stChatInput"] {
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        background-color: #18181B !important; /* Deep dark premium card */
        padding: 8px 12px 42px 12px !important; /* Extra bottom space for the toolbar row */
        box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
        position: relative !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        font-size: 0.88rem !important;
        color: #E2E8F0 !important;
        box-shadow: none !important;
        resize: none !important;
        min-height: 48px !important;
    }
    
    div[data-testid="stChatInput"] textarea:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Reposition circular send button inside the card */
    div[data-testid="stChatInput"] button {
        position: absolute !important;
        bottom: 8px !important;
        right: 12px !important;
        background-color: #2563EB !important; /* Premium Blue */
        border-radius: 50% !important;
        width: 28px !important;
        height: 28px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        transition: background-color 0.15s ease !important;
        padding: 0 !important;
        z-index: 101 !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background-color: #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")

GOOGLE_MODELS = [
    ("gemini-3.1-flash-lite", "Flash Lite · Fast & free-tier friendly"),
    ("gemini-3.5-flash", "Flash · Balanced quality"),
    ("gemini-2.5-flash-lite", "2.5 Flash Lite · Legacy"),
    ("gemini-2.5-flash", "2.5 Flash · Legacy"),
]
OLLAMA_MODELS = [
    ("llama3.2:3b", "Llama 3.2 3B · Best quality local"),
    ("qwen3.5:4b", "Qwen 3.5 4B · Balanced local"),
    ("qwen3.5:9b", "Qwen 3.5 9B · High quality local"),
    ("gemma4:e4b", "Gemma 4 E4B · Google local"),
]

# Token limits per model (approximate, for budget display)
MODEL_TOKEN_LIMITS = {
    "gemini-3.1-flash-lite": 1_000_000,
    "gemini-3.5-flash": 1_000_000,
    "gemini-2.5-flash-lite": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
}

# The 4 Agent Tools (get_current_date_time removed completely)
TOOLS = [
    ("🔍 search_documents", "tb-search", "Searches ingested files for relevant facts & citation sources"),
    ("🧮 calculator", "tb-calc", "Evaluates math expressions (e.g. '50000 * 0.15') dynamically when requested"),
    ("🌐 web_search", "tb-web", "Searches DuckDuckGo for live web facts if not found in files"),
    ("📝 summarize_document_topic", "tb-summary", "Summarizes topics or synthesis themes across files"),
]

TOOL_BADGE_MAP = {
    "search_documents": "tb-search",
    "calculator": "tb-calc",
    "web_search": "tb-web",
    "summarize_document_topic": "tb-summary",
}

# ──────────────────────────────────────────────────────────────────────────────
# Token estimation (≈4 chars per token)
# ──────────────────────────────────────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

# ──────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "provider": "google",
    "model_name": GOOGLE_MODELS[0][0],
    "session_tokens": 0,       # tokens used this session
    "total_tokens": 0,         # tokens used all-time this run
    "concise_mode": False,     # smart token saving
    "web_search_mode": False,  # force web search
    "selected_docs": set(),    # set of doc ids selected
    "selected_doc_names": set(), # set of selected doc filenames
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────
def _api_get(path: str, **kwargs):
    try:
        return requests.get(f"{BACKEND_API_URL}{path}", timeout=10, **kwargs)
    except requests.exceptions.RequestException:
        return None

def _api_post(path: str, **kwargs):
    try:
        return requests.post(f"{BACKEND_API_URL}{path}", timeout=180, **kwargs)
    except requests.exceptions.RequestException:
        return None

def _api_delete(path: str, **kwargs):
    try:
        return requests.delete(f"{BACKEND_API_URL}{path}", timeout=15, **kwargs)
    except requests.exceptions.RequestException:
        return None

def _tool_badge(tool_name: str) -> str:
    cls = TOOL_BADGE_MAP.get(tool_name, "tb-search")
    icons = {
        "search_documents": "🔍",
        "calculator": "🧮",
        "web_search": "🌐",
        "summarize_document_topic": "📝",
    }
    icon = icons.get(tool_name, "🔧")
    return f'<span class="tool-badge {cls}">{icon} {tool_name}</span>'

def _render_reasoning(reasoning: list):
    if not reasoning:
        return
    with st.expander(f"🔍 Agent Reasoning — {len(reasoning)} step(s)", expanded=False):
        for idx, step in enumerate(reasoning):
            tool = step.get("tool", "unknown")
            inp = step.get("tool_input", {})
            out = step.get("output", "")
            # Skip get_current_date_time rendering if present
            if tool == "get_current_date_time":
                continue
            st.markdown(
                f"""<div class="reasoning-step">
                    <strong>Step {idx+1}</strong> &nbsp;{_tool_badge(tool)}<br/>
                    <span style="color:#475569;font-size:0.76rem;">Input: <code>{inp}</code></span>
                    <div style="color:#94A3B8;white-space:pre-wrap;margin-top:6px;padding-top:6px;
                                border-top:1px dashed #1E293B;">{out}</div>
                </div>""",
                unsafe_allow_html=True,
            )

def _track_tokens(user_msg: str, assistant_msg: str):
    t = estimate_tokens(user_msg) + estimate_tokens(assistant_msg)
    st.session_state["session_tokens"] += t
    st.session_state["total_tokens"] += t

# ──────────────────────────────────────────────────────────────────────────────
# Fetch data from backend on run
# ──────────────────────────────────────────────────────────────────────────────
doc_resp = _api_get("/documents")
backend_online = doc_resp is not None
docs_list = doc_resp.json() if backend_online and doc_resp.status_code == 200 else []
doc_count = len(docs_list)

sessions_resp = _api_get("/chat/sessions")
sessions_list = sessions_resp.json() if backend_online and sessions_resp.status_code == 200 else []

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Smart Document Assistant")
    st.caption(
        "☁️ Google Gemini" if st.session_state["provider"] == "google" else "🖥️ Local Ollama"
    )
    st.divider()

    # ── MODEL CONFIGURATION ────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ Model & Provider</div>', unsafe_allow_html=True)

    provider = st.radio(
        "Provider",
        options=["google", "ollama"],
        index=0 if st.session_state["provider"] == "google" else 1,
        horizontal=True,
        help="**Google**: Cloud Gemini API · **Ollama**: Local models",
        label_visibility="collapsed",
    )
    st.session_state["provider"] = provider

    if provider == "google":
        st.markdown('<span class="provider-badge pb-google">☁️ Google Cloud</span>', unsafe_allow_html=True)
        model_options = [m[0] for m in GOOGLE_MODELS]
        model_labels  = [m[1] for m in GOOGLE_MODELS]
        default_idx = model_options.index(st.session_state["model_name"]) \
            if st.session_state["model_name"] in model_options else 0
        chosen = st.selectbox("Model", model_labels, index=default_idx, label_visibility="collapsed")
        model = model_options[model_labels.index(chosen)]
    else:
        st.markdown('<span class="provider-badge pb-ollama">🖥️ Ollama Local</span>', unsafe_allow_html=True)
        model_options = [m[0] for m in OLLAMA_MODELS]
        model_labels  = [m[1] for m in OLLAMA_MODELS]
        default_idx = model_options.index(st.session_state["model_name"]) \
            if st.session_state["model_name"] in model_options else 0
        chosen = st.selectbox("Model", model_labels, index=default_idx, label_visibility="collapsed")
        model = model_options[model_labels.index(chosen)]

    st.session_state["model_name"] = model

    # Smart token saving option
    if provider == "google":
        st.session_state["concise_mode"] = st.toggle(
            "🪙 Concise Mode (saves tokens)",
            value=st.session_state.get("concise_mode", False),
            help="Appends a brevity instruction to the agent input, cutting response lengths and token usage.",
        )
    else:
        st.session_state["concise_mode"] = False

    st.divider()

    # ── TOKEN USAGE ────────────────────────────────────────────────────────
    if provider == "google":
        st.markdown('<div class="section-header">📊 Token Usage (Est.)</div>', unsafe_allow_html=True)
        limit = MODEL_TOKEN_LIMITS.get(model, 1_000_000)
        sess_t = st.session_state["session_tokens"]
        pct = min(sess_t / limit * 100, 100)
        bar_class = "token-bar-warn" if pct > 70 else ""
        st.markdown(
            f"Session: **{sess_t:,}** / {limit:,} tokens"
            f'<div class="token-bar-bg"><div class="token-bar-fill {bar_class}" '
            f'style="width:{pct:.1f}%;"></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Total this run: {st.session_state['total_tokens']:,} tokens")
        if st.button("🔁 Reset Token Counter", use_container_width=True):
            st.session_state["session_tokens"] = 0
            st.toast("Token counter reset!", icon="🔁")
        st.divider()

    # ── DOCUMENT UPLOAD ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📁 Upload Documents</div>', unsafe_allow_html=True)
    if not backend_online:
        st.caption("Waiting for backend connectivity…")
    else:
        uploaded_file = st.file_uploader(
            "PDF or TXT",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            help="Supported: .pdf, .txt"
        )
        if uploaded_file:
            if st.button("⬆️ Ingest Document", use_container_width=True, type="primary"):
                with st.spinner("Processing & indexing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = _api_post("/documents/upload", files=files)
                    if resp is None:
                        st.error("Upload failed: API unreachable.")
                    elif resp.status_code == 201:
                        st.toast("✅ Ingested successfully!", icon="🎉")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', 'Unknown error')}")

    st.divider()

    # ── INGESTED DOCUMENTS REGISTRY & DELETION ──────────────────────────────
    st.markdown('<div class="section-header">📚 Ingested Documents</div>', unsafe_allow_html=True)
    if not backend_online:
        st.caption("🔄 Service starting up, please wait...")
    elif not docs_list:
        st.info("No documents uploaded yet.")
    else:
        if "doc_page" not in st.session_state:
            st.session_state["doc_page"] = 0
            
        doc_page_size = 5
        total_docs = len(docs_list)
        total_doc_pages = math.ceil(total_docs / doc_page_size)
        
        if st.session_state["doc_page"] >= total_doc_pages:
            st.session_state["doc_page"] = max(0, total_doc_pages - 1)
            
        current_doc_page = st.session_state["doc_page"]
        start_idx = current_doc_page * doc_page_size
        end_idx = start_idx + doc_page_size
        page_docs = docs_list[start_idx:end_idx]

        # Select All / Clear Selection Action Buttons
        col_all1, col_all2 = st.columns(2)
        with col_all1:
            if st.button("✓ Select All", use_container_width=True, key="btn_select_all_docs"):
                st.session_state["selected_docs"] = {doc["id"] for doc in docs_list}
                st.session_state["selected_doc_names"] = {doc["filename"] for doc in docs_list}
                st.rerun()
        with col_all2:
            if st.button("✗ Clear All", use_container_width=True, key="btn_clear_all_docs"):
                st.session_state["selected_docs"] = set()
                st.session_state["selected_doc_names"] = set()
                st.rerun()

        for doc in page_docs:
            ts = utc_to_ist(doc.get("upload_timestamp", ""))
            fname = doc["filename"]
            doc_id = doc["id"]
            is_selected = doc_id in st.session_state["selected_docs"]
            card_cls = "doc-card selected" if is_selected else "doc-card"

            # Sidebar Document Card Layout with Select and Delete
            col_card, col_sel, col_del = st.columns([6, 1.5, 1.5])
            with col_card:
                st.markdown(
                    f'<div class="{card_cls}"><div class="doc-name" title="{fname}">📄 {fname}</div>'
                    f'<div class="doc-meta">🕐 {ts}</div></div>',
                    unsafe_allow_html=True,
                )
            with col_sel:
                label = "✓" if is_selected else "☰"
                if st.button(label, key=f"sel_{doc_id}", help="Toggle selection of this document"):
                    if is_selected:
                        st.session_state["selected_docs"].remove(doc_id)
                        st.session_state["selected_doc_names"].remove(fname)
                    else:
                        st.session_state["selected_docs"].add(doc_id)
                        st.session_state["selected_doc_names"].add(fname)
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_doc_{doc_id}", help=f"Delete '{fname}' permanently"):
                    with st.spinner("Deleting..."):
                        del_resp = _api_delete(f"/documents/{doc_id}")
                        if del_resp and del_resp.status_code == 200:
                            if is_selected:
                                st.session_state["selected_docs"].discard(doc_id)
                                st.session_state["selected_doc_names"].discard(fname)
                            st.toast(f"Deleted document: {fname}", icon="🗑️")
                            st.rerun()
                        else:
                            detail = del_resp.json().get("detail", "Error deleting") if del_resp else "API offline"
                            st.error(f"Failed: {detail}")

        # Sidebar Document Pagination Controls Row
        if total_doc_pages > 1:
            st.markdown('<div style="margin-top:6px;"></div>', unsafe_allow_html=True)
            col_doc_prev, col_doc_page_num, col_doc_next = st.columns([2, 5, 2])
            with col_doc_prev:
                if st.button("◀", key="doc_prev_btn", disabled=(current_doc_page == 0), use_container_width=True, help="Previous page"):
                    st.session_state["doc_page"] = current_doc_page - 1
                    st.rerun()
            with col_doc_page_num:
                st.markdown(
                    f'<div style="text-align:center; font-size:0.75rem; color:#94A3B8; line-height:28px; font-weight:500;">'
                    f'{current_doc_page + 1} / {total_doc_pages}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_doc_next:
                if st.button("▶", key="doc_next_btn", disabled=(current_doc_page == total_doc_pages - 1), use_container_width=True, help="Next page"):
                    st.session_state["doc_page"] = current_doc_page + 1
                    st.rerun()

        n_selected = len(st.session_state["selected_docs"])
        if n_selected > 0:
            st.success(f"Selected **{n_selected}** document(s)")

    st.divider()

    # ── CHAT HISTORY & SESSION DELETION ─────────────────────────────────────
    st.markdown('<div class="section-header">💬 Chat Sessions</div>', unsafe_allow_html=True)
    active_sess = st.session_state["session_id"]
    st.markdown(
        f'<div class="doc-meta">Active: <code>{active_sess[:12]}…</code></div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.session_state["session_tokens"] = 0
            st.session_state["selected_docs"] = set()
            st.session_state["selected_doc_names"] = set()
            st.session_state["web_search_mode"] = False
            st.toast("New chat started!", icon="✨")
            st.rerun()
    with c2:
        if st.button("🔄 Reload", use_container_width=True, help="Sync sessions list"):
            st.rerun()

    # Stored Sessions list from DB
    if backend_online and sessions_list:
        st.markdown('<div style="font-size:0.75rem;color:#475569;margin-bottom:4px;">Previous Sessions:</div>', unsafe_allow_html=True)
        
        if "session_page" not in st.session_state:
            st.session_state["session_page"] = 0
            
        page_size = 5
        total_sessions = len(sessions_list)
        total_pages = math.ceil(total_sessions / page_size)
        
        if st.session_state["session_page"] >= total_pages:
            st.session_state["session_page"] = max(0, total_pages - 1)
            
        current_page = st.session_state["session_page"]
        start_idx = current_page * page_size
        end_idx = start_idx + page_size
        page_sessions = sessions_list[start_idx:end_idx]

        for s in page_sessions:
            s_id = s["id"]
            count = s["message_count"]
            time_str = utc_to_ist(s.get("created_at", ""))
            is_active = (s_id == active_sess)
            cls_active = "session-card active" if is_active else "session-card"

            col_sess_info, col_sess_load, col_sess_del = st.columns([6, 1.5, 1.5])
            with col_sess_info:
                disp_title = s.get("title") or f"Session {s_id[:8]}…"
                st.markdown(
                    f'<div class="{cls_active}">'
                    f'<div class="session-title" title="{s_id}">{disp_title}</div>'
                    f'<div class="session-meta">{count} messages · {time_str}</div></div>',
                    unsafe_allow_html=True,
                )
            with col_sess_load:
                if st.button("📂", key=f"load_s_{s_id}", help="Load this session history"):
                    hist_resp = _api_get(f"/chat/{s_id}/history")
                    if hist_resp and hist_resp.status_code == 200:
                        hist_data = hist_resp.json()
                        st.session_state["session_id"] = s_id
                        st.session_state["messages"] = [
                            {"role": m["role"], "content": m["content"], "reasoning": []}
                            for m in hist_data
                        ]
                        # estimate session tokens from loaded history
                        loaded_toks = sum(estimate_tokens(m["content"]) for m in hist_data)
                        st.session_state["session_tokens"] = loaded_toks
                        st.toast("Chat history loaded!", icon="📂")
                        st.rerun()
                    else:
                        st.error("Failed to load chat history.")
            with col_sess_del:
                if st.button("🗑️", key=f"del_s_{s_id}", help="Delete this session history permanently"):
                    del_s_resp = _api_delete(f"/chat/sessions/{s_id}")
                    if del_s_resp and del_s_resp.status_code == 200:
                        st.toast("Deleted session successfully", icon="🗑️")
                        if is_active:
                            st.session_state["session_id"] = str(uuid.uuid4())
                            st.session_state["messages"] = []
                            st.session_state["session_tokens"] = 0
                            st.session_state["selected_docs"] = set()
                            st.session_state["selected_doc_names"] = set()
                        st.rerun()
                    else:
                        st.error("Error deleting session")

        # Sidebar Pagination Controls Row
        if total_pages > 1:
            st.markdown('<div style="margin-top:6px;"></div>', unsafe_allow_html=True)
            col_prev, col_page_num, col_next = st.columns([2, 5, 2])
            with col_prev:
                if st.button("◀", key="sess_prev_btn", disabled=(current_page == 0), use_container_width=True, help="Previous page"):
                    st.session_state["session_page"] = current_page - 1
                    st.rerun()
            with col_page_num:
                st.markdown(
                    f'<div style="text-align:center; font-size:0.75rem; color:#94A3B8; line-height:28px; font-weight:500;">'
                    f'{current_page + 1} / {total_pages}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_next:
                if st.button("▶", key="sess_next_btn", disabled=(current_page == total_pages - 1), use_container_width=True, help="Next page"):
                    st.session_state["session_page"] = current_page + 1
                    st.rerun()
    elif backend_online:
        st.caption("No saved conversations in DB.")

    st.divider()

    # ── AGENT TOOLS INFO ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🛠️ Agent Tools (4)</div>', unsafe_allow_html=True)
    for name, badge_cls, desc in TOOLS:
        st.markdown(
            f'<div class="tool-info-row">'
            f'<span class="tool-badge {badge_cls}" style="margin-right:6px;">{name}</span>'
            f'<span class="tool-info-desc">{desc}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption("Agent uses these tools automatically depending on your input query.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────────────────────────────────────
# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 💬 Smart Document Assistant")
    provider_str = st.session_state["provider"].capitalize()
    model_str = st.session_state["model_name"]
    st.caption(f"Provider: **{provider_str}** · Model: `{model_str}`")
with col_h2:
    if backend_online:
        st.success("🟢 Connected to Backend", icon=None)
    else:
        st.error("🔴 Backend Offline", icon=None)

# Metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{doc_count}</div>'
        '<div class="metric-label">Ingested Files</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{len(st.session_state["messages"]) // 2}</div>'
        '<div class="metric-label">Exchanges</div></div>', unsafe_allow_html=True)
with m3:
    tok_display = f"{st.session_state['session_tokens']:,}"
    st.markdown(
        f'<div class="metric-card"><div class="metric-value" style="font-size:1.3rem;">{tok_display}</div>'
        '<div class="metric-label">Session Tokens (est.)</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{len(TOOLS)}</div>'
        '<div class="metric-label">Agent Tools Active</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── Chat Input Toolbar (Floating inside input box) ───────────────────────────
triggered_prompt = None
# ── Chat History ────────────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("reasoning"):
            _render_reasoning(msg["reasoning"])

st.markdown('<div class="toolbar-anchor"></div>', unsafe_allow_html=True)
col_tb1, col_tb2 = st.columns([3, 1])

with col_tb1:
    active_model = st.session_state["model_name"]
    provider_emoji = "🤖" if st.session_state["provider"] == "google" else "💻"
    st.markdown(f'<div class="model-badge-chat">{provider_emoji} {active_model}</div>', unsafe_allow_html=True)

with col_tb2:
    selected_doc_names = list(st.session_state.get("selected_doc_names", []))
    if selected_doc_names:
        doc_label = f"{len(selected_doc_names)} files" if len(selected_doc_names) > 1 else selected_doc_names[0]
        if st.button(f"📝 Summarize: {doc_label[:12]}…", key="tb_summarize_btn", help="Generate summary of selected file(s)"):
            triggered_prompt = (
                f"Provide a detailed, structured summary of the selected document(s): {', '.join(selected_doc_names)}. "
                f"Cover the main topics, key findings, and any important figures or conclusions."
            )
    else:
        st.button("📝 Summarize", key="tb_summarize_btn_disabled", disabled=True, help="Select document(s) in sidebar first")

    web_mode = st.session_state.get("web_search_mode", False)
    if st.button(
        "🌐 Web: ON" if web_mode else "🌐 Web Search",
        key="tb_web_search_btn",
        help="Toggle live internet search mode"
    ):
        st.session_state["web_search_mode"] = not web_mode
        st.rerun()

# ── Chat Input ─────────────────────────────────────────────────────────────
prompt_text = triggered_prompt or st.chat_input(
    "Ask a question about your documents, request a calculation, or search the web…"
)

if prompt_text:
    user_display = prompt_text
    # Augment prompt if web search is enabled
    if st.session_state.get("web_search_mode") and not triggered_prompt:
        prompt_text = (
            f"Please search the web for current information about the following: {prompt_text}. "
            f"Use the web_search tool to find up-to-date facts and provide a concise answer with sources."
        )
        st.session_state["web_search_mode"] = False  # Reset after single request

    # Concise Mode
    if st.session_state["concise_mode"]:
        prompt_text += " (Please be concise and to the point in your response.)"

    # Display user query
    st.session_state["messages"].append({"role": "user", "content": user_display})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_display)

    # Call backend API
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent is thinking…"):
            payload = {
                "session_id": st.session_state["session_id"],
                "message": prompt_text,
                "provider": st.session_state["provider"],
                "model_name": st.session_state["model_name"],
                "selected_doc_names": list(st.session_state.get("selected_doc_names", [])),
            }
            resp = _api_post("/chat", json=payload)

        if resp is None:
            err = "⚠️ Backend API is unreachable. Please ensure the server is running."
            st.warning(err)
            st.session_state["messages"].append(
                {"role": "assistant", "content": err, "reasoning": []}
            )
        elif resp.status_code == 200:
            data = resp.json()
            output = data.get("output", "")
            reasoning = data.get("reasoning_trace", [])

            st.markdown(output)
            st.session_state["messages"].append(
                {"role": "assistant", "content": output, "reasoning": reasoning}
            )
            _render_reasoning(reasoning)
            _track_tokens(user_display, output)

            # Reset document selection after summarizing
            if triggered_prompt:
                st.session_state["selected_docs"] = set()
                st.session_state["selected_doc_names"] = set()
                st.rerun()
        else:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            # Rate limit checking
            if "429" in str(detail) or "RESOURCE_EXHAUSTED" in str(detail):
                err = (
                    "⚠️ **API Rate Limit Reached.**\n\n"
                    "You've hit the free-tier quota for the current model. "
                    "Options:\n"
                    "- Switch to **gemini-3.1-flash-lite** in the sidebar (higher free quota)\n"
                    "- Wait ~1 minute before retrying\n"
                    "- Switch to **Ollama** for unlimited local inference"
                )
            else:
                err = f"❌ Error: {str(detail)[:500]}"
            st.warning(err)
            st.session_state["messages"].append(
                {"role": "assistant", "content": err, "reasoning": []}
            )
