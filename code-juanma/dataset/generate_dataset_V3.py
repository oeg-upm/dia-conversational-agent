import json
import random
import requests
import chromadb
import warnings
from typing import List, Literal
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
    "model": "llama3.1:8b",
    "base_url": "http://100.114.130.128:5000/v1",
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

    generation_method: Literal["llm_generated"] = Field(
        default="llm_generated", 
        description="Method used for generation"
    )

    language: str = Field(
        description="ISO language code (e.g., 'es', 'en', 'it')",
        pattern=r"^[a-z]{2}$" 
    )
    
    question: str = Field(description="Student query.")
    answer: str = Field(default="", description="Real RAG system response.")
    ground_truth: str = Field(description="Ideal concise answer.")
    
    contexts: List[str] = Field(description="Chunks retrieved at inference.")
    reference_contexts: List[str] = Field(description="Chunks used to generate ground truth.")
    
    source_document: str = Field(description="Filename of the source PDF.")
    chunk_id: str = Field(description="Unique identifier for the chunk.")
    
    question_type: str = Field(description="factual, procedural, comparative, out_of_scope or ambiguous")

    topic: Literal["plan_de_estudios", "matricula", "tfm", "profesorado", "otros"] = Field(
        description="Thematic area"
    )

    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Difficulty based on synthesis and implicitness"
    )

# ==========================================
# 3. UTILS & RAG API CALL
# ==========================================

def get_rag_response(question: str, chunk_metadata: dict) -> dict:
    """Sends the question to the real RAG backend using the new hierarchical context."""
    try:
        # Extract metadata for ContextItem
        course = chunk_metadata.get("course", "Unknown")
        degree = chunk_metadata.get("degree", "Unknown")
        source = chunk_metadata.get("source", "Unknown")

        # Build the payload
        payload = {
            "message": question,
            "selected_context": [
                {
                    "course": course,
                    "degree": degree,
                    "source": source
                }
            ],
            "chat_history": [] # Empty history for evaluation
        }

        
        response = requests.post(f"{BACKEND_URL}/chat?context=True", json=payload)
        response.raise_for_status()
        data = response.json()
        
        return {
            "response": data.get("response", "No response from RAG"),
            "context": data.get("context", [])
        }

    except Exception as e:
        return {"response": f"RAG Error: {str(e)}", "context": []}

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
    print(f"---- Total chunks in DB: {len(chunks)}")
    if not chunks: return
    
    # Shuffle chunks to get a diverse sample if n < total
    random.shuffle(chunks)
    selected_chunks = chunks[:n]
    
    llm = ChatOpenAI(**LLM_CONFIG)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an academic evaluator for RAG systems. Your task is to generate high-quality QA pairs in Spanish.

        STRICT FORMAT RULES:
        1. language: Use ONLY two-letter ISO codes (e.g., 'es', 'en').
        2. generation_method: Always use 'llm_generated'.
        3. topic: Categorize into: plan_de_estudios, matricula, tfm, profesorado, or otros.
        4. difficulty: 
           - 'easy': Answer is explicitly stated in a single sentence.
           - 'medium': Requires consulting multiple parts of the text or minor paraphrasing.
           - 'hard': Answer is implicit, requires synthesis of multiple sources, or the question is open-ended.

        Taxonomy of question_type:
        - factual: Single-hop fact.
        - procedural: Step-by-step process.
        - comparative: Synthesis of information.
        - out_of_scope: Plausible but missing from text.
        - ambiguous: Vague query.
        """),
        ("human", "Context: {chunk_text}\nMetadata: {metadata}\nType requested: '{q_type}'")
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
            res = get_rag_response(record.question, chunk['metadata'])
            record.answer = res['response']
            record.contexts = res['context']

            
            dataset.append(record.model_dump())
        except Exception as e:
            print(f"   Error in loop: {e}")

    return dataset

if __name__ == "__main__":
    # Generate 20 samples
    data = generate_evaluation_dataset(100)
    if data:
        output_file = "rag_dataset_v3.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Done! Dataset saved to {output_file}")