import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    # Step 1: Load the API key
    project_folder = Path(__file__).resolve().parent
    load_dotenv(project_folder / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    # Step 2: Prepare the texts
    texts = [
        "Python is a programming language.",
        "Streamlit is used to build web apps.",
        "Dogs are friendly animals.",
        "Cats like to sleep a lot."
    ]

    # Step 3: Initialize the OpenAI embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Step 4: Load an existing store or create and save a new one
    store_folder = project_folder / "my_store"
    index_file = store_folder / "index.faiss"
    metadata_file = store_folder / "index.pkl"

    if index_file.exists() and metadata_file.exists():
        # Load only a trusted store that you created yourself
        vector_store = FAISS.load_local(
            str(store_folder),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Existing FAISS vector store loaded!")
    else:
        vector_store = FAISS.from_texts(texts, embeddings)
        print("All texts are stored in the vector store!")

        vector_store.save_local(str(store_folder))
        print("FAISS vector store saved!")

    # Step 5: Search by meaning
    query = "Tell me about pets"

    # k=2 returns the top two matching texts
    results = vector_store.similarity_search(query, k=2)

    print(f"\nTop matches for: {query}")

    for result in results:
        print("-", result.page_content)


if __name__ == "__main__":
    main()