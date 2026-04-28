import requests
from typing import List
from .config import settings

def embed_text(text: str) -> List[float]:
    """
    Uses Ollama /api/embed (recommended).
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/embed"
    r = requests.post(url, json={"model": settings.OLLAMA_EMBED_MODEL, "input": text}, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Spec: returns {"embeddings":[...]} or {"embedding":[...]} depending on version/model.
    if "embeddings" in data and isinstance(data["embeddings"], list):
        # Some versions return list of embeddings for list input; for single input still list.
        # If it's nested, flatten first element.
        if len(data["embeddings"]) > 0 and isinstance(data["embeddings"][0], list):
            return data["embeddings"][0]
        return data["embeddings"]
    if "embedding" in data:
        return data["embedding"]
    raise ValueError(f"Unexpected embed response: {data}")

def generate(prompt: str) -> str:
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": settings.OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")