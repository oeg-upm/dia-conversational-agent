import httpx
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="Qwen/Qwen3.5-9B",
    base_url=f"http://100.112.68.72:8000/v1",
    api_key="not_required"
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