import os
import shutil
from typing import List, Optional, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

from backend.rag import BasicRAG

app = FastAPI(title="RAG DIA")
rag = BasicRAG()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tracks unique file IDs already ingested so re-uploads are skipped
processed_files: set = set()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ContextItem(BaseModel):
    course: str
    degree: str
    source: str


class ChatRequest(BaseModel):
    message: Optional[str] = None
    question: Optional[str] = None
    selected_context: Optional[List[ContextItem]] = None
    selected_files: Optional[List[str]] = None
    chat_history: List[Any] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    course: Optional[str] = Form("Unknown"),
    category: Optional[str] = Form("Unknown"),
    degree: Optional[str] = Form("Unknown"),
):
    global processed_files

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved_paths = []

    for file_obj in files:
        filename = file_obj.filename
        temp_path = os.path.join(UPLOAD_DIR, filename)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file_obj.file, buffer)

        saved_paths.append(temp_path)

    status_message = rag.add_documents_from_files(
        file_paths=saved_paths,
        course=course,
        category=category,
        degree=degree,
        processed_files=processed_files,
    )

    for path in saved_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    return {"processed_files": list(processed_files), "status_message": status_message}


@app.post("/chat")
async def chat(request: ChatRequest, k: int = 6, context: bool = False):
    """
    Query endpoint. Accepts both legacy and advanced payload shapes.
    - k: number of chunks to retrieve (for H6 experiment).
    - context: if True, returns the retrieved chunks alongside the answer.
    """
    question = request.message or request.question
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'message' or 'question' field")

    if request.selected_context:
        selected = [ctx.model_dump() for ctx in request.selected_context]
    elif request.selected_files:
        selected = request.selected_files
    else:
        return {"answer": "No context selected.", "response": "No context selected."}

    answer = await rag.query(question, selected, k=k)

    if context:
        return {
            "answer": answer,
            "response": answer,
            "context": [doc.page_content for doc in rag.last_retrieved_docs],
        }

    return {"answer": answer, "response": answer}


@app.get("/list_documents")
async def list_documents():
    return rag.list_documents()


@app.get("/files")
async def get_available_files():
    data = rag.list_documents()
    return {"hierarchy": data["hierarchy"]}


@app.get("/inspector")
async def inspector():
    html = rag.get_inspector_html()
    return {"html": html}