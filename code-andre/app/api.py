import time
import uuid
from fastapi import FastAPI
from pydantic import BaseModel

from .retriever import retrieve
from .prompting import build_prompt
from .ollama_client import generate
from .logging_ import log_event
from .config import settings

app = FastAPI(title="Local RAG API (stateless)")

class QueryIn(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(q: QueryIn):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    retrieved = retrieve(q.question, top_k=settings.TOP_K)
    prompt = build_prompt(q.question, retrieved)
    answer = generate(prompt)

    latency_ms = int((time.time() - t0) * 1000)

    sources = []
    for r in retrieved:
        meta = r["metadata"]
        sources.append({
            "doc_id": meta.get("doc_id"),
            "chunk_id": f"{meta.get('doc_id')}:{meta.get('chunk_index')}",
            "filename": meta.get("filename"),
            "distance": r.get("distance"),
        })

    log_event({
        "request_id": request_id,
        "question": q.question,
        "top_k": settings.TOP_K,
        "retrieved": sources,
        "answer": answer,
        "latency_ms": latency_ms,
        "models": {
            "llm": settings.OLLAMA_LLM_MODEL,
            "embed": settings.OLLAMA_EMBED_MODEL
        }
    })

    return {
        "request_id": request_id,
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms
    }