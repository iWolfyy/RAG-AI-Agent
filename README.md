# ⚡ Cerebras Elite RAG Agent

A high-performance, event-driven Retrieval-Augmented Generation (RAG) agent designed for speed and accuracy. Powered by the **Cerebras Wafer-Scale Engine (WSE)** for instant inference, **Google Gemini** for state-of-the-art embeddings, and **Inngest** for reliable background processing.

![Banner](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.14%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Key Features

- **Extreme Speed**: Leverages **Cerebras Llama 3.3 70B** for sub-second responses even with large contexts.
- **Durable Ingestion**: Decoupled document processing pipeline using **Inngest**, ensuring PDFs are parsed, chunked, and embedded reliably in the background without freezing the UI.
- **Advanced Search**: Semantic retrieval using **Google Gemini (`text-embedding-004`)** and **Qdrant** vector database.
- **Premium UI**: specific **Streamlit** interface with a custom "Glassmorphism" design system, dark mode, and fluid animations.
- **Document Management**: Drag-and-drop PDF upload with automatic indexing and session management.

---

## 🛠️ Tech Stack

### AI Core

- **Inference**: [Cerebras Cloud](https://cerebras.ai/) (Llama 3.3 70B)
- **Embeddings**: [Google GenAI](https://ai.google.dev/) (Gemini `text-embedding-004`)

### Backend & Data

- **Framework**: FastAPI
- **Orchestration**: [Inngest](https://www.inngest.com/) (Event-driven queues & flows)
- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **Data Loading**: LlamaIndex (PDF parsing)

### Frontend

- **Interface**: Streamlit
- **Styling**: Custom CSS (Inter font, Glassmorphism, Animated Gradients)

---

## 🏗️ Architecture Overview

1.  **Ingestion Flow**:

    - User uploads PDF in Streamlit.
    - File is saved locally.
    - Event `rag/ingest_pdf` is sent to Inngest.
    - **Inngest Worker** picks up the job:
      1.  Parses PDF with LlamaIndex.
      2.  Chunks text.
      3.  Embeds with Google Gemini.
      4.  Upserts to Qdrant.

2.  **Query Flow**:
    - User asks a question.
    - System embeds the query (Gemini).
    - Searches Qdrant for top-k relevant chunks.
    - Constructs prompt with context.
    - Streams answer from Cerebras (Llama 3.3).

---

## ⚡ Getting Started

### Prerequisites

- Python 3.12+
- Node.js (for Inngest CLI)
- Docker (for Qdrant local)
- `uv` (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-ai-agent.git
cd rag-ai-agent
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
# AI Providers
CEREBRAS_API_KEY=your_cerebras_key
GEMINI_API_KEY=your_google_ai_key

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Leave empty for local

# Inngest (Optional for local dev, required for prod)
INNGEST_EVENT_KEY=local
INNGEST_SIGNING_KEY=local
```

### 3. Start Infrastructure

Run Qdrant using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Install Dependencies

Using `uv`:

```bash
uv sync
```

### 5. Run the Application

You need three terminal windows to run the full stack:

**Terminal 1: Backend (FastAPI + Inngest Worker)**

```bash
uv run uvicorn main:app
```

**Terminal 2: Inngest Dev Server**

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
```

**Terminal 3: Frontend (Streamlit)**

```bash
uv run streamlit run streamlit_app.py
```

---

## 📂 Project Structure

```
├── main.py              # FastAPI app & Inngest function definitions (Backend)
├── streamlit_app.py     # Main UI application (Frontend)
├── vector_db.py         # Qdrant client wrapper & search logic
├── data_loader.py       # PDF parsing & Google Gemini embedding logic
├── custom_types.py      # Pydantic models for data validation
├── .env                 # Environment variables
├── pyproject.toml       # Dependencies (uv)
└── README.md            # Documentation
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

_Built with ❤️ using Cerebras, Inngest, and Streamlit._
