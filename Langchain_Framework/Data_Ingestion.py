# Create a sample file to load
text = """LangChain is a framework for building AI apps.
It helps connect language models with your own data.
RAG means Retrieval-Augmented Generation."""

with open("sample.txt", "w") as f:
    f.write(text)

print("sample.txt file created!")


from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
documents = loader.load()

print("Number of documents:", len(documents))
print("Content:")
print(documents[0].page_content)


from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_path = (
    Path.home()
    / "Downloads"
    / "50_Powerful_Prompt_Keywords.pdf"
)

if not pdf_path.is_file():
    print(f"PDF not found: {pdf_path}")
    print("Check the filename and confirm it is in Downloads.")
else:
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"PDF loaded: {pdf_path.name}")
    print(f"Number of pages: {len(documents)}")

    for page_number, document in enumerate(documents, start=1):
        print(f"\n--- Page {page_number} ---")
        print(document.page_content)