import streamlit as st
import os
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

# --- 1. FAST IMPORTS & CACHING ---
# We cache the clients so they don't recreate connections on every rerun
from cerebras.cloud.sdk import Cerebras
import inngest

load_dotenv()

@st.cache_resource
def init_engine():
    """Initializes heavy resources once and keeps them in memory."""
    from vector_db import QdrantStorage
    from data_loader import embed_texts
    
    # Initialize Clients
    cerebras = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
    qdrant = QdrantStorage()
    inngest_cli = inngest.Inngest(app_id="rag_app", is_production=False)
    
    return cerebras, qdrant, inngest_cli, embed_texts

# Boot the engine
cerebras_client, qdrant_storage, inngest_client, embed_fn = init_engine()

# --- 2. PROFESSIONAL STYLING ---
st.set_page_config(page_title="Cerebras Elite RAG", page_icon="⚡", layout="wide")

# Custom CSS for a sleek, dark-mode 'Glassmorphism' look
st.markdown("""
    <style>
    /* Global Settings & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif; 
        color: #e0e0e0;
    }
    
    /* Background & Main Container */
    .stApp {
        background-color: #0B0F19; /* Deep Space Blue */
        background-image: 
            radial-gradient(at 10% 10%, rgba(76, 29, 149, 0.2) 0px, transparent 50%),
            radial-gradient(at 90% 90%, rgba(37, 99, 235, 0.2) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 4px 0 24px rgba(0,0,0,0.4);
    }
    
    /* Document Uploader Card */
    .upload-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .upload-card:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(59, 130, 246, 0.8);
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1.5px;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 12px rgba(96, 165, 250, 0.3);
    }
    
    .sub-header {
        color: #94A3B8;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }

    /* Chat Messages */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 1.5rem 0 !important;
    }
    
    /* User Message Bubble */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #3B82F6 !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-radius: 12px;
        padding: 1rem !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }

    /* Assistant Message Bubble */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #8B5CF6 !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(139, 92, 246, 0.1) !important;
        border-radius: 12px;
        padding: 1rem !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
    }

    /* Input Box */
    .stChatInput {
        position: fixed;
        bottom: 2rem;
        width: 100%;
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(45deg, #1f6feb 0%, #58a6ff 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.4);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #ef4444 0%, #f87171 100%);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    </style>
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">⚡ Cerebras Elite RAG</h1>
        <p class="sub-header">Instant Knowledge Retrieval • Powered by Llama 3.3 70B</p>
    </div>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# --- 4. SIDEBAR: DOCUMENT CONTROL ---
with st.sidebar:
    st.image("https://cerebras.ai/wp-content/uploads/2024/01/cropped-favicon-270x270.png", width=60)
    st.title("Knowledge Hub")
    
    uploaded = st.file_uploader("Drop a PDF here", type="pdf", label_visibility="collapsed")
    
    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if file_id not in st.session_state.ingested_files:
            with st.status("💎 Indexing Document...", expanded=False) as status:
                # Local persistence
                uploads_dir = Path("uploads")
                uploads_dir.mkdir(exist_ok=True)
                file_path = uploads_dir / uploaded.name
                file_path.write_bytes(uploaded.getbuffer())
                
                # Hand off to Inngest for background worker execution
                asyncio.run(inngest_client.send(inngest.Event(
                    name="rag/ingest_pdf",
                    data={"pdf_path": str(file_path.resolve()), "source_id": uploaded.name}
                )))
                
                st.session_state.ingested_files.add(file_id)
                status.update(label="Handed off to Background Worker!", state="complete")
            st.toast(f"Processing {uploaded.name}...", icon="🚀")

    st.divider()
    
    # Database Management
    st.subheader("Database Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔥 Wipe DB", use_container_width=True, type="primary"):
            qdrant_storage.wipe_database()
            st.session_state.ingested_files = set()
            st.rerun()

# --- 5. MAIN CHAT: INSTANT STREAMING ---
# Display historical context
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input & Logic
if prompt := st.chat_input("Ask a question about your documents..."):
    # Immediate User Feedback
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Response Generation
    with st.chat_message("assistant"):
        # Phase 1: Semantic Search
        with st.status("🔍 Searching Knowledge Base...", expanded=False) as search_status:
            q_vec = embed_fn([prompt])[0]
            search_results = qdrant_storage.search(q_vec, top_k=5)
            context_text = "\n\n".join(search_results["contexts"])
            search_status.update(label=f"Found {len(search_results['contexts'])} relevant chunks", state="complete")

        # Phase 2: Instant Streaming with Cerebras
        response_container = st.empty()
        full_text = ""
        
        # We call Cerebras directly here for sub-second first-token latency
        stream = cerebras_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Use the provided context to answer the user. Be concise."},
                {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {prompt}"}
            ],
            model="llama3.3-70b",
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_text += token
            response_container.markdown(full_text + "▌")
        
        response_container.markdown(full_text)

        # Phase 3: Display Sources
        if search_results["sources"]:
            with st.expander("📚 Sources Referenced"):
                for src in set(search_results["sources"]):
                    st.markdown(f"- `{src}`")

    # Finalize state
    st.session_state.messages.append({"role": "assistant", "content": full_text})