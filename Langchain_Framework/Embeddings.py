from langchain_huggingface import HuggingFaceEmbeddings

# A small, free, popular embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding model is ready!")



from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# A small, cheap, high-quality OpenAI embedding model
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key = api_key)

print("OpenAI embedding model is ready!")



from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

text = "I love programming in Python."

# Step 1: Generate the original embedding
vector = embeddings.embed_query(text)

# Step 2: Find the minimum and maximum values
minimum = min(vector)
maximum = max(vector)

# Step 3: Rescale the entire vector to the range 0.5–1.0
if maximum == minimum:
    display_values = [0.75] * len(vector)
else:
    display_values = [
        0.5 + 0.5 * (value - minimum) / (maximum - minimum)
        for value in vector
    ]

# Step 4: Print the results
print("Text:", text)
print("Number of values:", len(vector))
print("Original first 5:", vector[:5])
print("Rescaled first 5:", [round(x, 6) for x in display_values[:5]])
print("Rescaled minimum:", min(display_values))
print("Rescaled maximum:", max(display_values))



import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Create the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


def similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# Generate embeddings
v1 = embeddings.embed_query("I like dogs")
v2 = embeddings.embed_query("I love puppies")
v3 = embeddings.embed_query("The car is fast")

# Compare sentences
print("dogs vs puppies (similar):", round(similarity(v1, v2), 2))
print("dogs vs car (different):", round(similarity(v1, v3), 2))