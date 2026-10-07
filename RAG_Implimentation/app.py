import hashlib

import streamlit as st

from config import (
    PDF_PATH,
    INDEX_DIR,
    EMBEDDING_MODEL,
    OPENAI_MODEL,
    SEARCH_K,
    TOP_K,
    RRF_K,
    BM25_WEIGHT,
    FAISS_WEIGHT,
)

from rag_chain import (
    create_rag_resources,
    ask_question,
)


st.set_page_config(
    page_title="Infrastructure Hybrid RAG",
    page_icon="📚",
    layout="wide",
)

st.title("Infrastructure Hybrid RAG Assistant")

st.caption(
    "BM25 keyword search + FAISS semantic search "
    "+ OpenAI answers + LangSmith tracing"
)


with st.sidebar:
    st.header("Configuration")

    st.write(f"PDF: {PDF_PATH.name}")
    st.caption(f"Embeddings: {EMBEDDING_MODEL}")
    st.caption(f"OpenAI: {OPENAI_MODEL}")

    st.write(f"Candidates per search: {SEARCH_K}")
    st.write(f"Final passages: {TOP_K}")

    st.info(
        "After updating the PDF, stop the app, "
        "run python ingest.py, then restart."
    )

    if st.button("Reload search indexes"):
        st.session_state.pop("resources", None)
        st.session_state.pop("result", None)

    st.caption(
        "Questions and retrieved passages go to OpenAI. "
        "Enabled traces go to LangSmith."
    )


try:
    documents_path = INDEX_DIR / "documents.json"
    index_path = INDEX_DIR / "index.faiss"

    if (
        not PDF_PATH.is_file()
        or not documents_path.is_file()
        or not index_path.is_file()
    ):
        raise FileNotFoundError(
            "PDF or index missing. "
            "Check PDF_PATH and run python ingest.py."
        )

    identity = (
        str(PDF_PATH),
        hashlib.sha256(PDF_PATH.read_bytes()).hexdigest(),
        documents_path.stat().st_mtime_ns,
        index_path.stat().st_mtime_ns,
        EMBEDDING_MODEL,
        OPENAI_MODEL,
        SEARCH_K,
        TOP_K,
        RRF_K,
        BM25_WEIGHT,
        FAISS_WEIGHT,
    )

    if st.session_state.get("identity") != identity:
        st.session_state.pop("resources", None)
        st.session_state.pop("result", None)

    if "resources" not in st.session_state:
        with st.spinner(
            "Loading FAISS and building BM25 keyword search..."
        ):
            st.session_state["resources"] = (
                create_rag_resources()
            )

            st.session_state["identity"] = identity

    resources = st.session_state["resources"]

except Exception as error:
    st.error(str(error))
    st.stop()


st.success(
    f"Ready: {resources['chunk_count']} chunks "
    "available for BM25 and FAISS."
)


with st.form("question_form"):
    question = st.text_input(
        "Ask about your infrastructure PDF",
        placeholder="How are EKS and Helm used?",
        max_chars=2000,
    )

    submitted = st.form_submit_button(
        "Ask",
        type="primary",
    )


if submitted:
    st.session_state.pop("result", None)

    if not question.strip():
        st.warning("Enter a question.")

    else:
        try:
            with st.spinner(
                "Searching with BM25 and FAISS, "
                "then generating an answer..."
            ):
                st.session_state["result"] = ask_question(
                    question,
                    resources,
                )

        except Exception as error:
            st.error(
                f"Request failed: {error}"
            )


if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader("Answer")
    st.markdown(result["answer"])

    column1, column2, column3 = st.columns(3)

    column1.metric(
        "BM25 candidates",
        result["bm25_count"],
    )

    column2.metric(
        "FAISS candidates",
        result["faiss_count"],
    )

    column3.metric(
        "Combined passages",
        len(result["sources"]),
    )

    st.subheader("Retrieved passages")

    st.caption(
        "RRF scores indicate ranking, not answer confidence. "
        "Inspect passages to verify citations."
    )

    for source in result["sources"]:
        title = (
            f"[{source['label']}] "
            f"{source['source']} — "
            f"PDF page {source['page']}"
        )

        with st.expander(title):
            st.caption(
                "Retrieved by: "
                + ", ".join(source["retrieved_by"])
            )

            st.caption(
                f"RRF score: {source['rrf_score']:.6f}"
            )

            st.write(source["text"])