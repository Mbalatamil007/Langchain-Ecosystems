import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

pdf_setting = os.getenv("PDF_PATH", "").strip()

PDF_PATH = (
    Path(pdf_setting)
    if pdf_setting
    else (
        BASE_DIR
        / "data"
        / "FINAL_Enterprise_Infrastructure_Architecture_Project_Knowledge.pdf"
    )
)

if not PDF_PATH.is_absolute():
    PDF_PATH = BASE_DIR / PDF_PATH

INDEX_DIR = BASE_DIR / "vector_db" / "faiss_index"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini",
)

CHUNK_SIZE = 650
CHUNK_OVERLAP = 100

# Candidate passages retrieved by each search method.
SEARCH_K = 10

# Passages sent to OpenAI after fusion.
TOP_K = 5

# Reciprocal Rank Fusion settings.
RRF_K = 60
BM25_WEIGHT = 0.5
FAISS_WEIGHT = 0.5


def validate_api_keys():
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError(
            "Set OPENAI_API_KEY in .env and restart Streamlit."
        )

    tracing_enabled = (
        os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )

    if tracing_enabled:
        if not os.getenv("LANGSMITH_API_KEY", "").strip():
            raise ValueError(
                "Set LANGSMITH_API_KEY or set "
                "LANGSMITH_TRACING=false."
            )