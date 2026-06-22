"""
generate_base_dataset.py
------------------------
Phase 1 of the H6 experiment.
Generates questions and ground truths from ChromaDB chunks using the LLM.
Does NOT call the RAG backend — answers and contexts are left empty.
Run this once, then use experiment_h6.py to generate answers for each k.
"""

import json
import os
import random
import warnings
from typing import List, Literal

from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ==========================================
# CONFIGURATION
# ==========================================

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_dia"

LLM_CONFIG = {
    "model": "qwen2.5:32b",
    "base_url": "http://100.80.246.115:5000/v1",
    "api_key": "not_required",
    "temperature": 0.7
}

OUTPUT_FILE = "datasets/base_dataset.json"
N_SAMPLES = 50  # number of QA pairs to generate

# ==========================================
# DATA SCHEMA
# ==========================================

class QAPair(BaseModel):
    model_config = ConfigDict(extra='ignore')

    sample_id: str = Field(description="Consecutive ID.")
    generation_method: Literal["llm_generated"] = Field(default="llm_generated")
    language: str = Field(description="ISO language code", pattern=r"^[a-z]{2}$")
    question: str = Field(description="Student query.")
    answer: str = Field(default="", description="Filled later by experiment_h6.py")
    ground_truth: str = Field(description="Ideal concise answer.")
    contexts: List[str] = Field(default=[], description="Filled later by experiment_h6.py")
    reference_contexts: List[str] = Field(description="Chunks used to generate ground truth.")
    source_document: str = Field(description="Filename of the source PDF.")
    chunk_id: str = Field(description="Unique identifier for the chunk.")
    question_type: Literal["factual", "summarization", "multi_hop", "ambiguous"]


# ==========================================
# HELPERS
# ==========================================

def get_db_chunks():
    """Fetches all chunks from ChromaDB."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection(COLLECTION_NAME)
        res = collection.get()
        if not res['ids']:
            print("Collection is empty.")
            return []
        return [{
            "id": res['ids'][i],
            "text": res['documents'][i],
            "metadata": res['metadatas'][i]
        } for i in range(len(res['ids']))]
    except Exception as e:
        print(f"Connection failed: {e}")
        return []

# ==========================================
# GENERATION
# ==========================================

def generate_base_dataset(n: int = N_SAMPLES) -> list:
    chunks = get_db_chunks()
    print(f"Total chunks in DB: {len(chunks)}")
    if not chunks:
        return []

    random.shuffle(chunks)
    selected_chunks = chunks[:n]

    llm = ChatOpenAI(**LLM_CONFIG)


    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un evaluador académico para sistemas RAG. Tu tarea es generar pares de pregunta-respuesta de alta calidad en español.

        REGLAS DE FORMATO ESTRICTAS:
        1. language: Usar siempre 'es'.
        2. generation_method: Usar siempre 'llm_generated'.
        3. answer: Dejar SIEMPRE este campo como cadena vacía "". NO rellenarlo.
        4. contexts: Dejar SIEMPRE este campo como lista vacía []. NO rellenarlo.

        Taxonomía de question_type:
        - factual: Hecho de un solo paso explícitamente indicado en el texto.
        - summarization: Requiere sintetizar o agrupar múltiples elementos de información.
        - multi_hop: Requiere combinar información de diferentes partes del documento para inferir la respuesta.
        - ambiguous: Consulta vaga o imprecisa.

        REGLAS CRÍTICAS PARA ground_truth:
        - Escribe una respuesta COMPLETA y AUTOCONTENIDA en español que responda plenamente a la pregunta.
        - La respuesta debe ser suficientemente informativa para que alguien sin acceso al documento fuente pueda entenderla.
        - Debe contener todos los hechos clave necesarios para evaluar una respuesta RAG (fechas, porcentajes, nombres, condiciones).
        - NUNCA copies códigos en bruto ni texto de tablas del contexto (p.ej. '14, 3 = ...' o 'ASI Natura 103000361').
        - NUNCA uses 'sí', 'no' o palabras sueltas como respuesta.
        - Escribe siempre en prosa continua. NUNCA uses bullet points, listas numeradas ni markdown.
        - Buen ejemplo: 'La asignatura se evalúa mediante dos prácticas grupales con peso del 30% cada una y un examen final del 40%.'
        - Mal ejemplo: '14, 3 = Presentation of second assignment' o 'sí' o 'ASI Natura 103000361'
        """),
        ("human", "Contexto: {chunk_text}\nMetadatos: {metadata}\nTipo solicitado: '{q_type}'")
    ])

    generator = prompt | llm.with_structured_output(QAPair)


    q_types = ["factual", "summarization", "multi_hop", "ambiguous"]
    probabilities = [0.5, 0.2, 0.2, 0.1]

    dataset = []

    for i, chunk in enumerate(selected_chunks):
        q_type = random.choices(q_types, weights=probabilities, k=1)[0]
        print(f"[{i+1}/{len(selected_chunks)}] Generating {q_type}...")

        try:
            record = generator.invoke({
                "chunk_text": chunk['text'],
                "metadata": json.dumps(chunk['metadata']),
                "q_type": q_type
            })

            record.sample_id = str(i + 1)

            # Force empty regardless of what the LLM filled
            record.answer = ""
            record.contexts = []

            if q_type == "Unanswerable":
                record.source_document = "N/A"
                record.reference_contexts = []
                record.chunk_id = "N/A"
                record.ground_truth = "Unanswerable"  # force label
            else:
                record.source_document = chunk['metadata'].get('source', 'unknown')
                record.chunk_id = chunk['id']
                record.reference_contexts = [chunk['text']]

            # Store chunk metadata for Phase 2 (experiment_h6.py)
            item = record.model_dump()
            item['chunk_metadata'] = chunk['metadata']
            dataset.append(item)

        except Exception as e:
            print(f"  Error: {e}")

    return dataset


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1: Generating base dataset (questions + ground truth)")
    print("=" * 60)

    data = generate_base_dataset(N_SAMPLES)

    if data:
        os.makedirs("datasets", exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nDone! Base dataset saved to {OUTPUT_FILE} ({len(data)} items)")
    else:
        print("No data generated.")