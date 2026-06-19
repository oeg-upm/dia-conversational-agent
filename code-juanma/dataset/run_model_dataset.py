import json
import requests
import time
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

BACKEND_URL = "http://localhost:8001"

INPUT_FILE = "rag_dataset_v3_qwen2.5_32b.json" 

OUTPUT_FILE = "rag_dataset_v3_no_mq_llama_3.1_8b_instruct_few_shot.json"

HIERARCHY_CACHE = None

COLLECTION = "rag_collection"

# ==========================================
# RAG API CALL FUNCTION
# ==========================================

def get_all_sources_for_degree(course: str, degree: str) -> list:
    """Fetch all documents belonging to a specific course and degree from the backend."""
    global HIERARCHY_CACHE
    if HIERARCHY_CACHE is None:
        try:
            response = requests.get(f"{BACKEND_URL}/files?collection_name={COLLECTION}")
            response.raise_for_status()
            HIERARCHY_CACHE = response.json().get("hierarchy", {})
        except Exception as e:
            print(f"Warning: Could not fetch hierarchy from backend ({e})")
            HIERARCHY_CACHE = {}

    sources = []

    if course in HIERARCHY_CACHE and degree in HIERARCHY_CACHE[course]:
        for display_name in HIERARCHY_CACHE[course][degree]:
            raw_filename = display_name.split("] ", 1)[-1]
            sources.append(raw_filename)
            
    return sources

def get_rag_response(question: str, chunk_metadata: dict) -> dict:
    """Sends the question to the real RAG backend using the new hierarchical context."""
    try:
        # Extract metadata for ContextItem
        course = chunk_metadata.get("course", "Unknown")
        degree = chunk_metadata.get("degree", "Unknown")
        original_source = chunk_metadata.get("source", "Unknown")

        all_sources = get_all_sources_for_degree(course, degree)

        if not all_sources:
            all_sources = [original_source]

        selected_context = []

        for src in all_sources:
            selected_context.append({
                "course": course,
                "degree": degree,
                "source": src
            })

        # Build the payload
        payload = {
            "message": question,
            "selected_context": selected_context,
            "chat_history": [], # Empty history for evaluation
            "collection_name": COLLECTION
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
    print(f"Evaluating against collection: {COLLECTION}")
    
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    print(f"Processing {len(dataset)} questions...")

    for item in tqdm(dataset, desc="Querying RAG"):
        question = item.get("question")
        chunk_id = item.get("chunk_id", "")
        
        course = "Unknown" 
        degree = "Unknown"
        source = "Unknown"
        
        if chunk_id and chunk_id != "N/A":
            parts = chunk_id.split("_")
            
            if len(parts) >= 3:
                course = f"{parts[0]}_{parts[1]}"
                degree = parts[2]
                if len(parts) > 3:
                    source = "_".join(parts[3:]) 

        metadata_dict = {
            "course": course,
            "degree": degree,
            "source": source
        }
        
        rag_res = get_rag_response(question, metadata_dict)
        
        # Update the dictionary with the new results
        item["answer"] = rag_res["response"]
        item["contexts"] = rag_res["context"]

    # Save the new dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        
    print(f"\nDone! The new dataset has been saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate()