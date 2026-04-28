from typing import List, Dict, Any
from .vectorstore import get_collection
from .ollama_client import embed_text
from .config import settings

def retrieve(question: str, top_k: int = None) -> List[Dict[str, Any]]:
    col = get_collection()
    q_emb = embed_text(question)
    k = top_k or settings.TOP_K

    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "text": doc,
            "metadata": meta,
            "distance": dist
        })
    return out