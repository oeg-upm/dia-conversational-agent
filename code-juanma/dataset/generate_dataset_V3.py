import json
import random
import requests
import chromadb
import warnings
from typing import List
from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ==========================================
# 1. CONFIGURATION
# ==========================================

CHROMA_HOST = "localhost" 
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_collection"
BACKEND_URL = "http://localhost:8001"

LLM_CONFIG = {
    "model": "qwen2.5:32b",
    "base_url": "http://100.83.249.109:5000/v1",
    "api_key": "not_required",
    "temperature": 0.7
}

# ==========================================
# 2. DATA SCHEMA
# ==========================================

class QAPair(BaseModel):
    """Data schema for a single QA pair-"""
    model_config = ConfigDict(extra='ignore')
    
    sample_id: str = Field(description="Consecutive ID.")
    generation_method: str = Field(description="always 'llm_generated'")
    language: str = Field(description="es (spanish), en (english), etc.")
    
    question: str = Field(description="Student query.")
    answer: str = Field(default="", description="Real RAG system response.")
    ground_truth: str = Field(description="Ideal concise answer.")
    
    contexts: List[str] = Field(description="Chunks retrieved at inference.")
    reference_contexts: List[str] = Field(description="Chunks used to generate ground truth.")
    
    source_document: str = Field(description="Filename of the source PDF.")
    chunk_id: str = Field(description="Unique identifier for the chunk.")
    
    question_type: str = Field(description="factual, procedural, comparative, out_of_scope, ambiguous")
    topic: str = Field(description="Thematic area, 'plan_de_estudios', 'matricula', 'tfm', 'profesorado', etc.")
    difficulty: str = Field(description="easy, medium or hard")

# ==========================================
# 3. UTILS & RAG API CALL
# ==========================================

def get_rag_response(question: str, source_file: str) -> str:
    """Sends the question to the real RAG backend to get the answer."""
    try:
        # We send the specific file as selected_files to ensure context
        payload = {
            "message": question,
            "selected_files": [source_file] if source_file != "N/A" else []
        }
        response = requests.post(f"{BACKEND_URL}/qa_chat", json=payload)
        response.raise_for_status()
        data = response.json()
        response = data.get("response", "No response from RAG")
        context = data.get("context", "No context returned")

        return {"response": response, "context": context}


    except Exception as e:
        return f"RAG Error: {str(e)}"

def get_db_chunks():
    """Fetches chunks from ChromaDB."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection(COLLECTION_NAME)
        res = collection.get()
        
        if not res['ids']:
            print("---- Collection is empty.")
            return []
        
        indices = list(range(len(res['ids'])))
        return [{
            "id": res['ids'][i],
            "text": res['documents'][i],
            "metadata": res['metadatas'][i]
        } for i in indices]
    except Exception as e:
        print(f"---- Connection failed: {e}")
        return []

# ==========================================
# 4. DATASET GENERATION
# ==========================================

def generate_evaluation_dataset(n: int = 20):
    chunks = get_db_chunks()
    if not chunks: return
    
    # Shuffle chunks to get a diverse sample if n < total
    random.shuffle(chunks)
    selected_chunks = chunks[:n]
    
    llm = ChatOpenAI(**LLM_CONFIG)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an academic evaluator. Generate a QA pair in Spanish.
        Taxonomy:
        - factual: Single-hop fact.
        - procedural: Step-by-step process.
        - comparative: Synthesis of info.
        - out_of_scope: Plausible but missing from text.
        - ambiguous: Vague query.
        """),
        ("human", "Context: {chunk_text}\nMetadata: {metadata}\nType: '{q_type}'")
    ])

    generator = prompt | llm.with_structured_output(QAPair)
    
    dataset = []
    # Probability distribution for question types
    q_types = ["factual", "procedural", "comparative", "out_of_scope", "ambiguous"]
    probabilities = [0.4, 0.2, 0.2, 0.1, 0.1]

    for i, chunk in enumerate(selected_chunks):
        # 1. Select question type based on probability
        q_type = random.choices(q_types, weights=probabilities, k=1)[0]
        
        print(f"---- [{i+1}/{len(selected_chunks)}] Generating {q_type}...")
        
        try:
            # 2. Generate question and ground truth
            record = generator.invoke({
                "chunk_text": chunk['text'],
                "metadata": json.dumps(chunk['metadata']),
                "q_type": q_type
            })
            
            # 3. Assign consecutive ID
            record.sample_id = str(i + 1)
            
            # 4. Enforce traceability metadata
            if q_type == "out_of_scope":
                record.source_document = "N/A"
                record.reference_contexts = []
                record.chunk_id = "N/A"
            else:
                record.source_document = chunk['metadata'].get('source', 'unknown')
                record.chunk_id = chunk['id']
                record.reference_contexts = [chunk['text']]
            
            record.contexts = [chunk['text']]
            
            # 5. Get RAG answer
            print(f"      > Fetching RAG response for: '{record.question[:50]}...'")
            res = get_rag_response(record.question, record.source_document)
            record.answer = res['response']
            record.contexts = res['context']

            
            dataset.append(record.model_dump())
        except Exception as e:
            print(f"   Error in loop: {e}")

    return dataset

if __name__ == "__main__":
    # Generate 20 samples
    data = generate_evaluation_dataset(20)
    if data:
        output_file = "rag_dataset_v3.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Done! Dataset saved to {output_file}")




# Falta el contexto, son los retrived chunks