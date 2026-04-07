import asyncio
import os
import shutil
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import chromadb



# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama import OllamaEmbeddings


# --- Docling Imports ---
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.chunking import HybridChunker

app = FastAPI(title="RAG DIA")

# --- 1. Initialization ---
print("Initializing FastAPI backend...")

ollama_url = "http://100.114.130.128:5000" 

# LLM Local / Cluster
llm = ChatOpenAI(
    model="qwen2.5:32b", 
    base_url=ollama_url + "/v1",
    api_key="not_required",
    temperature=0.1
)

# Embeddings (cluster)

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b",
    base_url=ollama_url
)

# ChromaDB client
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
vectorstore = Chroma(
    client=chroma_client,
    collection_name="rag_collection",
    embedding_function=embeddings
)

# --- Variables ---
processed_files = set()
LAST_RETRIEVED_DOCS = []
SESSION_HISTORY = []

# --- Pydantic model ---
class ContextItem(BaseModel):
    course: str
    degree: str
    source: str

class ChatRequest(BaseModel):
    message: str
    selected_context: List[ContextItem]
    chat_history: List[Any] = []


def reciprocal_rank_fusion(results: list[list], k=60):
    """RRF implementation to fuse results from multiple queries. Each result is a list of documents sorted by relevance."""
    fused_scores = {}
    doc_lookup = {}

    for docs in results:

        for rank, doc in enumerate(docs):

            source = doc.metadata.get("source", "unknown")
            chunk = doc.metadata.get("chunk_index", -1)

            # Unique ID
            doc_id = f"{source}_chunk_{chunk}"

            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_lookup[doc_id] = doc

            fused_scores[doc_id] += 1.0 / (rank + k)

    # Sort by fused score
    reranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return unique docuements with their fused scores
    return [
        (doc_lookup[doc_id], score)
        for doc_id, score in reranked
    ]


# --- 2. Endpoints ---

@app.get("/files")
async def get_available_files():
    """Returns hierarchy: Course -> Degree -> [Prefix] Filename."""
    try:
        # Get all metadata entries from the vectorstore
        db_data = vectorstore.get(include=["metadatas"])
        metadatas = db_data.get("metadatas", [])
        
        hierarchy = {}
        for meta in metadatas:
            if not meta: continue
            c = meta.get("course", "Unknown")
            g = meta.get("degree", "Unknown")
            s = meta.get("source", "Unknown")
            
            if c not in hierarchy: hierarchy[c] = {}
            if g not in hierarchy[c]: hierarchy[c][g] = set()
            
            # Add the filename with a prefix to indicate its degree (e.g., [Bachelor], [Master])
            display_name = f"[{c}] {s}"
            hierarchy[c][g].add(display_name)

        cleaned_hierarchy = {}
        for c, degrees in hierarchy.items():
            cleaned_hierarchy[c] = {g: sorted(list(files)) for g, files in degrees.items()}
                
        return {"hierarchy": cleaned_hierarchy}
    except Exception as e:
        print(f"Error retrieving hierarchy: {e}")
        return {"hierarchy": {}}
    

@app.post("/upload")
async def process_files(
    files: List[UploadFile] = File(...),
    course: Optional[str] = Form("Unknown"), # Default value to avoid breaking frontend
    category: Optional[str] = Form("Unknown"), # Degree/Master
    degree: Optional[str] = Form("Unknown")
):
    """
    Processes uploaded files and attaches hierarchical metadata.
    'Optional' and default values.
    """
    global processed_files
    new_docs = []
    new_filenames = []
    os.makedirs("temp_uploads", exist_ok=True)

    for file_obj in files:
        filename = file_obj.filename
        # Unique ID to prevent collisions between different years of the same subject
        unique_file_id = f"{course}_{degree}_{filename}"
        
        if unique_file_id in processed_files:
            continue 

        temp_path = os.path.join("temp_uploads", filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file_obj.file, buffer)

        try:
            print(f"Processing: {filename} | Course: {course} | Degree: {degree}")
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True 
            pipeline_options.ocr_options = EasyOcrOptions() 
            
            doc_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )

            # qwen tokenizer for better chunking consistency
            custom_chunker = HybridChunker(
                tokenizer="Qwen/Qwen2-0.5B",
                max_tokens=600,
                overlap_tokens=200,
                merge_peers=True
            )

            loader = DoclingLoader(
                file_path=temp_path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=custom_chunker,
                converter=doc_converter
            )
            
            splits = loader.load()

            for i, doc in enumerate(splits):
                # Metadata for filtering in ChromaDB
                doc.metadata["source"] = filename
                doc.metadata["course"] = course
                doc.metadata["category"] = category
                doc.metadata["degree"] = degree
                doc.metadata["chunk_index"] = i
                
                # Context injection for better interpretability of retrieved chunks
                doc.page_content = f"[{course} - {degree} - {filename}]\n{doc.page_content}"
            
            new_docs.extend(splits)
            new_filenames.append(filename)
            processed_files.add(unique_file_id)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if new_docs:
        cleaned_docs = filter_complex_metadata(new_docs)
        # Unique IDs to avoid overwriting chunks from different degrees with the same subject name
        doc_ids = [f"{doc.metadata['course']}_{doc.metadata['degree']}_{doc.metadata['source']}_ch_{doc.metadata['chunk_index']}" 
                   for doc in cleaned_docs]
        
        vectorstore.add_documents(documents=cleaned_docs, ids=doc_ids)
        status_message = f"Success: {len(cleaned_docs)} chunks added from {len(new_filenames)} files."
    else:
        status_message = "No new files were added (they might already exist)."

    return {"processed_files": list(processed_files), "status_message": status_message}


@app.post("/chat")
async def chat_response(request: ChatRequest, context: bool = False):
    global LAST_RETRIEVED_DOCS
    try:
        
        if not request.message: 
            raise HTTPException(status_code=400, detail="Empty message")
        
        selected_context = request.selected_context
        if not request.selected_context: 
            return {"response": "No context selected."}
        
        if len(selected_context) == 1:
            # If only one file is selected, we can directly filter by that source
            ctx = selected_context[0]
            search_filter = {
                "$and": [
                    {"course": ctx.course},
                    {"degree": ctx.degree},
                    {"source": ctx.source}
                ]
            }
        else:
            # If multiple files are selected, we create a filter that matches any of the selected contexts
            filter_list = []
            for ctx in selected_context:
                filter_list.append({
                    "$and": [
                        {"course": ctx.course},
                        {"degree": ctx.degree},
                        {"source": ctx.source}
                    ]
                })
            search_filter = {"$or": filter_list}
        
        formatted_history = ""
        for turn in SESSION_HISTORY:
            formatted_history += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
        
        if not formatted_history or context:
            formatted_history = "Beginning of the conversation."


        print(f"\n--- Starting RAG-Fusion for: '{request.message}' ---")

        # --- 1. Multi-Query Generation ---
        mq_template = """
            You are an expert Academic Search Assistant. Your goal is to rewrite and expand the user's 
            current question into 5 distinct, standalone search queries for a vector database.

            CRITICAL RULES:
            1. CONTEXTUAL RESOLUTION: If the user's question contains pronouns (it, they, he, she) or 
            implicit references (e.g., "and for the other one?", "what about the credits?"), 
            you MUST use the Chat History to resolve these references into full subject names or topics.
            2. STANDALONE QUERIES: Each generated query must be complete and understandable 
            WITHOUT the chat history. 
            3. PERSPECTIVES: Generate queries covering different aspects: formal name, 
            specific requirements, evaluation criteria, and related terminology.
            4. LANGUAGE: Always output the queries in the same language as the user's question.

            Chat history:
            {chat_history}

            User question:
            {question}

            Output only the 5 standalone alternative queries, one per line, no numbering.
        """

        prompt_mq = PromptTemplate.from_template(mq_template)
        mq_chain = prompt_mq | llm | StrOutputParser()
        
        generated_queries_str = mq_chain.invoke({
            "question": request.message, 
            "chat_history": formatted_history
        })


        queries = [request.message] + [q.strip() for q in generated_queries_str.split('\n') if q.strip()]
        print(f"Generated queries:\n{queries}")

        # --- 2. Parallel recovery ---
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 6, "filter": search_filter}
        )
        
        all_retrieved_results = []
        tasks = [retriever.ainvoke(q) for q in queries]
        all_retrieved_results = await asyncio.gather(*tasks)

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        fused_docs = reciprocal_rank_fusion(all_retrieved_results)
        
        final_top_docs = [doc for doc, score in fused_docs[:6]]
        LAST_RETRIEVED_DOCS = final_top_docs

        context_text = ""
        for d in final_top_docs:
            context_text += f"\n---\nFILE: {d.metadata.get('source')} | YEAR: {d.metadata.get('course')} | DEGREE: {d.metadata.get('degree')}\n"
            context_text += f"CONTENT: {d.page_content}\n"


        print(f"Final retrived chunks (after RRF):")
        for i, doc in enumerate(final_top_docs):
            print(f"  {i+1}. {doc.metadata.get('course')} - {doc.metadata.get('degree')} - {doc.metadata.get('source', 'Unknown')} - Chunk {doc.metadata.get('chunk_index', 0)}")

        template_qa = (
            "You are an expert Academic Advisor for university students.\n"
            "Your task is to answer the user's question using EXCLUSIVELY the provided context.\n\n"
            
            "STRICT RULES:\n"
            "1. NO EXTERNAL KNOWLEDGE: use only the provided fragments. If the context doesn't contain the answer, "
            "simply state that you don't know.\n"
            "2. CLARITY: be concise but clear. If the question is ambiguous, state that you don't understand and ask for clarification instead of guessing.\n"
            "3. STRUCTURE: use bullet points or numbered lists for complex information if it is needed (like evaluation criteria or syllabus).\n"
            "4. NO HALLUCINATIONS: do not invent dates, names of professors, or percentages if they are not explicitly in the context.\n"
            "5. LANGUAGE: respond in the same language as the user's question.\n\n"
            
            "CHAT HISTORY (for conversation flow):\n"
            "{chat_history}\n\n"
            
            "CONTEXT (relevant fragments from academic guides):\n"
            "{context}\n\n"
            
            "USER QUESTION:\n"
            "{question}\n\n"
            
            "ANSWER (precise, structured):"
        )
        
        prompt_qa = ChatPromptTemplate.from_template(template_qa)
        qa_chain = prompt_qa | llm | StrOutputParser()
        

        response = qa_chain.invoke({
            "context": context_text,
            "question": request.message,
            "chat_history": formatted_history
        })

        SESSION_HISTORY.append({
            "user": request.message,
            "bot": response
        })

        if len(SESSION_HISTORY) > 10:
            SESSION_HISTORY.pop(0)

        if context:
            return {"response": response, "context": [doc.page_content for doc in final_top_docs]}
        else:
            return {"response": response}
    
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return {"response": f"Error: {str(e)}"}


@app.get("/inspector")
async def visualize_extended_context():
    global LAST_RETRIEVED_DOCS
    if not LAST_RETRIEVED_DOCS:
        return {"html": "<div style='padding:20px; text-align:center;'>No context available. Please ask a question first.</div>"}

    html_output = f"<h3 style='margin-bottom:20px; color: #333;'>Context used for the last response</h3>"

    for i, doc in enumerate(LAST_RETRIEVED_DOCS):
        source = doc.metadata.get("source", "Unknown")
        course = doc.metadata.get("course", "Unknown")
        degree = doc.metadata.get("degree", "Unknown")
        idx = doc.metadata.get("chunk_index", 0)
        
        # Retrieve previous and next chunks for better context visualization
        prev_id = f"{course}_{degree}_{source}_ch_{idx - 1}"
        next_id = f"{course}_{degree}_{source}_ch_{idx + 1}"
        
        text_prev = "<i>(Start of document)</i>"
        text_next = "<i>(End of document)</i>"
        
        try:
            neighbors = vectorstore._collection.get(ids=[prev_id, next_id])
            if neighbors and "ids" in neighbors:
                for j, doc_id in enumerate(neighbors["ids"]):
                    if doc_id == prev_id: text_prev = neighbors["documents"][j]
                    elif doc_id == next_id: text_next = neighbors["documents"][j]
        except: pass

        html_output += f"""
        <div style="border: 1px solid #ccc; border-radius: 8px; margin-bottom: 30px; overflow: hidden; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background-color: white;">
            <div style="background-color: #e0e0e0; padding: 8px 15px; border-bottom: 1px solid #999; font-weight: bold; font-size: 0.9em; color: #000000;">
                Rank #{i+1} | {course} | {degree} | {source} | Chunk ID: {idx}
            </div>
            <div style="background-color: #fff3e0; padding: 10px; font-size: 0.85em; color: #444444; border-bottom: 1px dotted #ccc;">
                <strong style="color: #d84315;">Previous context:</strong><br>{text_prev}
            </div>
            <div style="background-color: #f1f8e9; padding: 15px; font-size: 1em; border-left: 5px solid #4caf50; color: #000000;">
                <strong style="color: #2e7d32;">Retrieved chunk:</strong><br>{doc.page_content}
            </div>
            <div style="background-color: #e3f2fd; padding: 10px; font-size: 0.85em; color: #444444; border-top: 1px dotted #ccc;">
                <strong style="color: #1565c0;">Next context:</strong><br>{text_next}
            </div>
        </div>
        """
    return {"html": html_output}