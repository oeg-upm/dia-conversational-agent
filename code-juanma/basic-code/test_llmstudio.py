import httpx
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen2.5:32b",
    base_url=f"http://100.80.173.80:5000/v1",
    api_key="sk-no-required"
)

messages = [
    (
        "system",
        "You are a helpful assistant that only translates English to Spanish.",
    ),
    ("human", "Pepe is passionate about his master's in Artificial Intelligence."),
]

try:
    aiMsg = model.invoke(messages)
    print("\nRespuesta del modelo:")
    print(aiMsg.content)
except Exception as e:
    print(f"Error: {e}")