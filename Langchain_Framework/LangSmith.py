import os

# Turn LangSmith tracking ON
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your_api_key"   # <-- paste your key
os.environ["LANGSMITH_PROJECT"] = "my-first-app"          # runs are grouped under this name
os.environ["OPENAI_API_KEY"] = "your_api_key"          # <-- paste your key

print("LangSmith is connected!")



from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()

# This run is automatically recorded in LangSmith.
print(chain.invoke({"topic": "LangSmith"}))


from langsmith import traceable

@traceable
def make_greeting(name):
    return f"Hello {name}, welcome to GenAI!"

# This call is now recorded in LangSmith as its own step.
print(make_greeting("Bala"))


from openai import OpenAI
from langsmith.wrappers import wrap_openai

# wrap_openai is the ONLY change needed for tracking.
client = wrap_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one word."}],
)

print(response.choices[0].message.content)