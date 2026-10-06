from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

texts = [
    "Python is a programming language used for AI.",
    "Streamlit is used to build web apps quickly.",
    "LangChain helps connect AI models with your data.",
    "Dogs are friendly and loyal animals."
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(texts, embeddings)

print("Vector store ready!")


retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # return top 2 results

print("Retriever is ready!")



question = "What is used to build web apps?"

results = retriever.invoke(question)

print("Question:", question)
print("\nRelevant information found:")
for r in results:
    print("-", r.page_content)