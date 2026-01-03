import streamlit as st
import os
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

# --- 1. CONFIGURATION & IMPORTS ---
st.set_page_config(
    page_title="Cerebras Elite RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Third-party imports with error handling
try:
    from cerebras.cloud.sdk import Cerebras
    import inngest
except ImportError as e:
    st.error(f"Failed to import required libraries: {e}. Please ensure all dependencies are installed.")
    st.stop()

# Load environment variables
load_dotenv()

# --- 2. ASSETS & ICONS (SVG) ---
def get_icon(name):
    """Returns SVG icons for a cleaner, professional look."""
    icons = {
        "cpu": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>""",
        "database": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>""",
        "trash": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>""",
        "refresh": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>""",
        "file-text": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>""",
        "upload": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>""",
        "zap": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>""",
        "alert-triangle": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>"""
    }
    return icons.get(name, "")

# --- 3. CORE ENGINE & RESOURCES ---
@st.cache_resource
def init_engine():
    """Initializes heavy resources once and keeps them in memory."""
    try:
        from vector_db import QdrantStorage
        from data_loader import embed_texts
        
        # Initialize Clients
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            st.warning("⚠️ CEREBRAS_API_KEY not found in environment variables. Functionality will be limited.")
            cerebras = None
        else:
            cerebras = Cerebras(api_key=api_key)
            
        qdrant = QdrantStorage()
        inngest_cli = inngest.Inngest(app_id="rag_app", is_production=False)
        
        return cerebras, qdrant, inngest_cli, embed_texts
    except Exception as e:
        st.error(f"Critical Error Initializing Engine: {e}")
        return None, None, None, None

# --- 4. UI STYLING & COMPONENTS ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* --- GLOBAL VARIABLES --- */
        :root {
            --primary-gradient: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
            --glass-bg: rgba(30, 41, 59, 0.4);
            --glass-border: rgba(255, 255, 255, 0.08);
            --skeleton-base: #1e293b;
            --skeleton-highlight: #334155;
        }

        /* --- TYPOGRAPHY & RESET --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #e2e8f0;
        }

        /* --- BACKGROUND --- */
        .stApp {
            background-color: #0B0F19;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.15), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(37, 99, 235, 0.15), transparent 25%);
            background-attachment: fixed;
        }

        /* --- SKELETON LOADER ANIMATION --- */
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        .skeleton-line {
            height: 12px;
            margin-bottom: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, 
                var(--skeleton-base) 25%, 
                var(--skeleton-highlight) 50%, 
                var(--skeleton-base) 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite linear;
        }
        .skeleton-text { width: 90%; }
        .skeleton-text-short { width: 60%; }

        /* --- WELCOME CARDS --- */
        .welcome-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .welcome-card {
            background: rgba(30, 41, 59, 0.4); /* Glass Effect */
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: left;
        }
        
        .welcome-card:hover {
            background: rgba(30, 41, 59, 0.6);
            border-color: #8B5CF6;
            transform: translateY(-4px);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }
        
        .card-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: rgba(59, 130, 246, 0.1);
            color: #60A5FA;
            margin-bottom: 1rem;
        }
        
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.5rem;
        }
        
        .card-desc {
            font-size: 0.9rem;
            color: #94A3B8;
        }

        /* --- SIDEBAR REFINEMENTS --- */
        section[data-testid="stSidebar"] {
            background-color: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--glass-border);
        }
        
        .sidebar-header {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        /* --- CHAT INTERFACE --- */
        .main .block-container { max-width: 900px; padding-top: 2rem; }

        /* Message Containers */
        [data-testid="stChatMessage"] {
            padding: 1rem;
            margin-bottom: 1.5rem;
            background: transparent;
        }
        
        /* User Bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.15);
            border-radius: 12px;
            margin-left: 10%;
        }
        
        /* Assistant Bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: transparent;
            margin-right: 5%;
        }
        
        /* Icons in Chat */
        [data-testid="chatAvatarIcon-user"] { background: #3B82F6 !important; }
        [data-testid="chatAvatarIcon-assistant"] { background: linear-gradient(135deg, #8B5CF6, #EC4899) !important; }

        /* --- INPUT FIELD --- */
        .stChatInput { border-radius: 16px !important; }
        [data-testid="stChatInput"] {
            background-color: rgba(30, 41, 59, 0.6) !important;
            border: 1px solid var(--glass-border) !important;
            color: white !important;
        }
        [data-testid="stChatInput"]:focus {
            border-color: #8B5CF6 !important;
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
        }

        /* --- UI UTILS --- */
        .icon-btn-container { display: flex; align-items: center; gap: 0.5rem; }
        
        /* Hiding Standard Elements */
        #MainMenu, footer { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)

def render_skeleton_loader():
    """Renders a shimmering skeleton loader for the AI response."""
    return """
        <div style="margin-top: 10px;">
            <div class="skeleton-line skeleton-text"></div>
            <div class="skeleton-line skeleton-text"></div>
            <div class="skeleton-line skeleton-text-short"></div>
        </div>
    """

def render_welcome_screen():
    st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem; margin-top: 2rem;">
            <h1 style="font-size: 3.5rem; background: linear-gradient(90deg, #FFFFFF, #94A3B8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -2px; margin-bottom: 0.5rem;">
                Cerebras Elite
            </h1>
            <p style="font-size: 1.2rem; color: #64748B;">
                What would you like to explore today?
            </p>
        </div>
        
        <div class="welcome-container">
            <div class="welcome-card" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'Summarize Document'}, '*')">
                <div class="card-icon">📄</div>
                <div class="card-title">Summarize Docs</div>
                <div class="card-desc">Get instant summaries from your uploaded PDFs</div>
            </div>
            <div class="welcome-card">
                <div class="card-icon">🧠</div>
                <div class="card-title">Deep Analysis</div>
                <div class="card-desc">Ask complex questions requiring semantic understanding</div>
            </div>
            <div class="welcome-card">
                <div class="card-icon">⚡</div>
                <div class="card-title">Fast Facts</div>
                <div class="card-desc">Retrieve specific data points in milliseconds</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 5. STATE MANAGEMENT ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = set()

# --- 6. MAIN LOGIC ---
def render_sidebar(qdrant_storage, inngest_client):
    with st.sidebar:
        # App Branding
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
                <div style="width: 32px; height: 32px; background: var(--primary-gradient); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white;">
                    {get_icon("zap")}
                </div>
                <div>
                    <h1 style="margin:0; font-size: 1.2rem; font-weight: 700;">Cerebras</h1>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Knowledge Base
        st.markdown(f'<div class="sidebar-header">{get_icon("database")} &nbsp; Knowledge Base</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        
        if uploaded:
            handle_file_upload(uploaded, inngest_client)
            
        if st.session_state.ingested_files:
            st.info(f"{len(st.session_state.ingested_files)} Files Indexed")
        else:
            st.caption("No documents in memory.")

        st.divider()

        # Actions
        st.markdown(f'<div class="sidebar-header">{get_icon("cpu")} &nbsp; System</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear CRT"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Wipe DB", type="primary"):
                if qdrant_storage:
                    qdrant_storage.wipe_database()
                    st.session_state.ingested_files = set()
                    st.toast("Database Wiped", icon="🔥")
                    time.sleep(1)
                    st.rerun()

def handle_file_upload(uploaded_file, inngest_client):
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if file_id in st.session_state.ingested_files:
        st.caption(f"✅ {uploaded_file.name} exists.")
        return

    try:
        with st.status("Indexing...", expanded=False) as status:
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            file_path = uploads_dir / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            
            if inngest_client:
                asyncio.run(inngest_client.send(inngest.Event(
                    name="rag/ingest_pdf",
                    data={"pdf_path": str(file_path.resolve()), "source_id": uploaded_file.name}
                )))
                st.session_state.ingested_files.add(file_id)
                status.update(label="Complete", state="complete")
    except Exception as e:
        st.error(f"Error: {e}")

def render_chat_interface(cerebras_client, qdrant_storage, embed_fn):
    # 1. Show Welcome Screen if empty
    if not st.session_state.messages:
        render_welcome_screen()

    # 2. History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. Input Loop
    if prompt := st.chat_input("Ask anything..."):
        if not cerebras_client:
            st.error("Engine Offline (Missing API Key)")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # A. SKELETON LOADER (Visual Feedback)
            skeleton_placeholder = st.empty()
            skeleton_placeholder.markdown(render_skeleton_loader(), unsafe_allow_html=True)
            
            try:
                # B. RETRIEVAL (Hidden Latency)
                context_text = ""
                sources = []
                if embed_fn and qdrant_storage:
                    try:
                        q_vec = embed_fn([prompt])[0]
                        search_results = qdrant_storage.search(q_vec, top_k=5)
                        context_text = "\n\n".join(search_results.get("contexts", []))
                        sources = search_results.get("sources", [])
                    except:
                        pass # Fallback to pure LLM

                # C. STREAMING (Typewriter Effect)
                # Remove skeleton before streaming starts
                skeleton_placeholder.empty()
                
                response_container = st.empty()
                full_text = ""
                
                stream = cerebras_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a precise AI assistant. Answer based on context."},
                        {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {prompt}"}
                    ],
                    model="llama3.3-70b",
                    stream=True
                )

                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    full_text += token
                    # The "▌" cursor adds the typewriter feel
                    response_container.markdown(full_text + "▌")
                
                # Finalize
                response_container.markdown(full_text)
                
                if sources:
                    with st.expander("📚 Sources"):
                        for src in set(sources):
                            st.caption(f"ref: {src}")

                st.session_state.messages.append({"role": "assistant", "content": full_text})

            except Exception as e:
                skeleton_placeholder.empty()
                st.error(f"Error: {e}")

# --- 7. MAIN APP ---
def main():
    inject_custom_css()
    init_session_state()
    cerebras, qdrant, inngest_cl, embed = init_engine()
    
    render_sidebar(qdrant, inngest_cl)
    render_chat_interface(cerebras, qdrant, embed)

if __name__ == "__main__":
    main()