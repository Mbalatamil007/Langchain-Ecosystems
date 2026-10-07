# Infrastructure Hybrid RAG Project

A PDF question-answering application using:

- LangChain
- Hugging Face embeddings
- FAISS semantic search
- BM25 keyword search through rank-bm25
- Reciprocal Rank Fusion
- OpenAI
- LangSmith
- Streamlit

## Application infrastructure

The application runs on your laptop.

PDF ingestion:
1. Read the configured PDF.
2. Extract page text.
3. Split text into overlapping chunks.
4. Embed chunks with Hugging Face.
5. Save FAISS vectors and JSON passages.

Application startup:
1. Load saved FAISS vectors and JSON passages.
2. Validate the PDF hash and embedding settings.
3. Build BM25 from the saved passages.
4. Create the OpenAI generation chain.

Question answering:
1. Run BM25 keyword search.
2. Run FAISS semantic search.
3. Combine rankings with Reciprocal Rank Fusion.
4. Select the final passages.
5. Generate an answer with OpenAI.
6. Display source passages and PDF page positions.
7. Record LangSmith traces when enabled.

## Components

| Component | Purpose |
|---|---|
| pypdf | PDF extraction |
| LangChain text splitter | Overlapping chunks |
| Hugging Face | Local embeddings |
| FAISS | Semantic retrieval |
| rank-bm25 | Keyword retrieval |
| RRF | Combine rankings and deduplicate |
| OpenAI | Grounded answer generation |
| LangSmith | Tracing |
| Streamlit | Interface |

## Project files

| File | Purpose |
|---|---|
| config.py | Paths and settings |
| embeddings.py | Embedding model |
| ingest.py | Build saved FAISS index |
| rag_chain.py | BM25, FAISS, fusion, and generation |
| app.py | Streamlit interface |
| requirements.txt | Dependencies |
| .env | Local secrets and settings |
| .env.example | Configuration template |
| .gitignore | Git exclusions |
| README.md | Instructions |

## PDF path

Set this in .env:

```dotenv
PDF_PATH=C:/Users/mbala/GenAI Program/RAG_Implimentation/FINAL_Enterprise_Infrastructure_Architecture_Project_Knowledge.pdf
```

The PDF can remain at this location.
The index is saved inside the project.

## Prerequisites

- 64-bit Python 3.11 or 3.12.
- OpenAI API key with API billing and model access.
- LangSmith API key if tracing is enabled.
- Internet for the first embedding-model download and API calls.

## Installation: Windows CMD

Run inside the folder containing app.py:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If .env does not exist:

```bat
copy .env.example .env
```

Add your API keys.

Set LANGSMITH_TRACING=false to disable tracing.
Use your LangSmith region endpoint and workspace ID if required.
Restart Streamlit after editing .env.

## Ingestion

```bat
python ingest.py
```

Generated files:

- vector_db/faiss_index/index.faiss
- vector_db/faiss_index/documents.json

Ingestion uses local embeddings and does not require an OpenAI key.
BM25 is rebuilt in memory from documents.json at application startup.

## Start the app

```bat
python -m streamlit run app.py
```

Open the local URL printed in the terminal.

Enter a question and click Ask.
Inspect the answer and expandable passages.
The interface shows whether each passage was retrieved by BM25,
FAISS, or both.

## Update the PDF

1. Stop Streamlit.
2. Replace the PDF or change PDF_PATH.
3. Run python ingest.py.
4. Restart Streamlit.

Rebuild after changing the embedding model or chunk settings.
Do not ingest while the application is serving requests.

## Retrieval settings

Edit config.py:

- SEARCH_K: candidates from each search method.
- TOP_K: final passages sent to OpenAI.
- BM25_WEIGHT: BM25 contribution.
- FAISS_WEIGHT: FAISS contribution.
- RRF_K: rank-fusion constant.

The default gives equal weight to BM25 and FAISS.

RRF combines rankings rather than incompatible raw similarity scores.
RRF scores are not confidence percentages.
No separate cross-encoder reranker is included.

## Example questions

- What does the PDF say about EKS and Helm?
- Explain the role of SSRS.
- How is Redis/ElastiCache used?
- How are application and database dependencies managed?
- What checks are required after deployment?

## LangSmith

Open the Infrastructure-Hybrid-RAG project.

Inspect:

- InfrastructureHybridRAG
- BM25KeywordSearch
- FAISSSemanticSearch
- ReciprocalRankFusion
- GenerateGroundedAnswer

## Flowchart images

The PNG images from the previous project version describe the
FAISS-only implementation.

They have not been regenerated for this hybrid version.
Use the application infrastructure steps above as the current workflow.

## Data handling

PDF extraction, embeddings, BM25, and FAISS run locally.

OpenAI receives the question and final retrieved passages.
LangSmith receives trace data when enabled.

OpenAI API usage is separate from a ChatGPT subscription.

## Limitations

- Scanned PDFs require OCR.
- BM25 uses case-insensitive word tokenization.
- Keyword matching does not provide stemming or synonym expansion.
- Retrieval can return irrelevant passages.
- Answers and citations require verification.
- Page positions start at 1 and may differ from printed page numbers.
- Questions are independent; conversation memory is not included.
- Authentication and production deployment are not included.
- Dependency ranges are not a tested exact version lock.
- This hybrid code has not been executed in the current environment.