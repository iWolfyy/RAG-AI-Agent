import os
from google import genai
from cerebras.cloud.sdk import Cerebras   # Use this for google-genai SDK
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()


# Configure the Gemini SDK
google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "text-embedding-004"
EMBED_DIM = 768  # Gemini default for text-embedding-004

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []

    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks

 
def embed_texts(texts: list[str]) -> list[list[float]]:
    # FIXED: Use client.models.embed_content for the new SDK
    response = google_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config={"task_type": "RETRIEVAL_DOCUMENT"}
    )
    # FIXED: The new SDK returns embeddings as a list of objects with a .values attribute
    return [item.values for item in response.embeddings]
