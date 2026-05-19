import json
import requests
import time
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

BACKEND_URL = "http://localhost:8001"

INPUT_FILE = "rag_dataset_v3_bge_qwen2.5_V2.json" 

OUTPUT_FILE = "rag_dataset_v3_bge_no_multiquery_qwen3.6.json" 

# ==========================================
# RAG API CALL FUNCTION
# ==========================================

def get_rag_response(question: str, source_doc: str, course: str, degree: str) -> dict:
    """Sends the question to the RAG backend and retrieves the response and new context."""
    try:
        payload = {
            "message": question,
            "selected_context": [
                {
                    "course": course, 
                    "degree": degree,
                    "source": source_doc
                }
            ],
            "chat_history": [] 
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

# ==========================================
# MAIN LOGIC
# ==========================================

def evaluate():
    print(f"Loading original dataset from: {INPUT_FILE}")
    
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    print(f"Processing {len(dataset)} questions...")

    for item in tqdm(dataset, desc="Querying RAG"):
        question = item.get("question")
        source_doc = item.get("source_document", "Unknown")
        chunk_id = item.get("chunk_id", "")

        course = "Unknown" 
        degree = "Unknown"
        
        if chunk_id and chunk_id != "N/A":
            parts = chunk_id.split("_")
            
            if len(parts) >= 3:
                course = f"{parts[0]}_{parts[1]}"
                degree = parts[2]
        
        # Make the request to RAG
        rag_res = get_rag_response(question, source_doc, course, degree)
        
        # Update the dictionary with the new results
        item["answer"] = rag_res["response"]
        item["contexts"] = rag_res["context"]

    # Save the new dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        
    print(f"\nDone! The new dataset has been saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate()