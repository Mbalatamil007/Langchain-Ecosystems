import hashlib
import json
import uuid
from io import BytesIO

from config import (
    PDF_PATH,
    INDEX_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

import faiss
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from embeddings import get_embeddings


def ingest_pdf():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    pdf_bytes = PDF_PATH.read_bytes()
    reader = PdfReader(BytesIO(pdf_bytes))

    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError(
            "Use an unencrypted PDF."
        )

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": PDF_PATH.name,
                        "page": page_number,
                    },
                )
            )

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "Scanned PDFs require OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(pages)

    document_ids = []

    for chunk_number, chunk in enumerate(chunks, start=1):
        document_id = str(uuid.uuid4())

        chunk.metadata["chunk_id"] = chunk_number
        chunk.metadata["document_id"] = document_id

        document_ids.append(document_id)

    print(f"PDF: {PDF_PATH}")
    print(f"Text pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print("Creating embeddings...")

    store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        ids=document_ids,
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        store.index,
        str(INDEX_DIR / "index.faiss"),
    )

    payload = {
        "embedding_model": EMBEDDING_MODEL,
        "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "documents": [
            {
                "id": document_id,
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
            for document_id, chunk in zip(document_ids, chunks)
        ],
    }

    (INDEX_DIR / "documents.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved FAISS index and documents to: {INDEX_DIR}")
    print("BM25 will use these saved documents at application startup.")


if __name__ == "__main__":
    ingest_pdf()