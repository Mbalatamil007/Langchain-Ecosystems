import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the API key from .env in the same folder as this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        f"OPENAI_API_KEY is missing. Check your .env file at: {env_path}"
    )

# 1. Prompt: a template with a blank {topic} to fill in
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one simple sentence."
)

# 2. Model: the OpenAI chat model
model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0
)

# 3. Output parser: turn the model's reply into plain text
output_parser = StrOutputParser()

print("All three pieces are ready!")

# 4. Combine the prompt, model, and output parser
chain = prompt | model | output_parser

print("Chain is ready!")

# 5. Run the explanation chain
answer = chain.invoke({"topic": "Python"})

print("\nExplanation:")
print(answer)

# 6. Create a second prompt for a beginner tip
tip_prompt = ChatPromptTemplate.from_template(
    "Give one short, friendly tip for a beginner learning {skill}."
)

# 7. Build and run the tip chain
tip_chain = tip_prompt | model | output_parser

tip = tip_chain.invoke({"skill": "Python"})

print("\nBeginner tip:")
print(tip)