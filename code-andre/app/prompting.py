from typing import List, Dict, Any

SYSTEM_RULES = """Eres un asistente RAG.
- Responde SOLO usando el contexto proporcionado.
- Si el contexto no alcanza para responder, di "No tengo suficiente información en los documentos indexados."
- Incluye una sección "Fuentes" con doc_id y chunk_id usados.
"""

def build_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    ctx_blocks = []
    used = []
    for r in retrieved:
        meta = r["metadata"]
        doc_id = meta.get("doc_id")
        chunk_index = meta.get("chunk_index")
        chunk_id = f"{doc_id}:{chunk_index}"
        used.append(chunk_id)
        ctx_blocks.append(f"[{chunk_id} | {meta.get('filename','')}] \n{r['text']}")

    context = "\n\n---\n\n".join(ctx_blocks)
    sources = ", ".join(used)

    prompt = f"""{SYSTEM_RULES}

Contexto:
{context}

Pregunta:
{question}

Instrucciones de salida:
1) Respuesta clara y directa (en español).
2) Fuentes: lista de chunk_id (doc_id:chunk_index) que usaste.

Fuentes disponibles:
{sources}
"""
    return prompt