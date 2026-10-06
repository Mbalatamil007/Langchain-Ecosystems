long_text = """Python is a programming language. It is easy to learn.
Streamlit is used to build web apps. LangChain helps build AI apps.
RAG combines search with AI models. Embeddings turn text into numbers.
Vector stores save those numbers. Retrievers find the best matches."""

print("Total characters:", len(long_text))


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,      # each chunk is about 80 characters
    chunk_overlap=10    # 10 characters repeat between chunks
)

chunks = splitter.split_text(long_text)

print("Number of chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)



from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = Path(
    r"C:\Users\mbala\OneDrive\Desktop\Documents for mine\Bala Resume\cover.txt"
)

if not file_path.is_file():
    print(f"File not found: {file_path}")
else:
    # Read the text file
    try:
        long_text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        long_text = file_path.read_text(encoding="cp1252")

    if not long_text.strip():
        print("The file is empty.")
    else:
        # Split text into chunks of up to 80 characters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=80,
            chunk_overlap=10
        )

        chunks = splitter.split_text(long_text)

        print("File:", file_path.name)
        print("Number of chunks:", len(chunks))

        for i, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {i}:")
            print(chunk)