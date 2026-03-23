import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Variables ---
print("Initializing system...")

# Local LLM (Llama 3.2 via LM Studio)
llm = ChatOpenAI(
    model="llama-3.2-3b-instruct",
    base_url="https://138.4.144.36/proxy/5000/llm",
    api_key="not_required",
    temperature=0.1
)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Vector database
vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name="basic_rag_collection"
)

# --- Data stores ---

# Track processed files and avoid duplicates
processed_files = set()

# Key: (filename, chunk_index) -> value: chunk text
GLOBAL_CHUNK_STORE = {} 

# Docs retrieved in the last chat interaction
LAST_RETRIEVED_DOCS = []

# --- 2. Backend functions ---

def process_files(files):
    """Loads files, splits them, indexes them, and stores raw text for visualization."""
    global GLOBAL_CHUNK_STORE, processed_files
    
    if not files:
        return gr.update(), "No files uploaded."

    new_docs = []
    new_filenames = []

    for file_obj in files:
        filename = os.path.basename(file_obj.name)
        
        # Skip if already processed
        if filename in processed_files:
            continue 

        try:
            # Load based on extension
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_obj.name)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_obj.name, encoding="utf-8")
            else:
                continue
            
            raw_docs = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(raw_docs)

            for i, doc in enumerate(splits):
                # Add metadata for neighbor lookup
                doc.metadata["source"] = filename
                doc.metadata["chunk_index"] = i
                
                # Store in global dictionary
                GLOBAL_CHUNK_STORE[(filename, i)] = doc.page_content
            
            new_docs.extend(splits)
            new_filenames.append(filename)
            processed_files.add(filename)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    status_message = ""
    if new_docs:
        vectorstore.add_documents(new_docs)
        status_message = f"Processed: {', '.join(new_filenames)}. Total chunks added: {len(new_docs)}"
    else:
        status_message = "No new valid files processed."

    # Update the CheckboxGroup with the list of files
    return gr.update(choices=list(processed_files), value=list(processed_files)), status_message

def chat_response(message, history, selected_files):
    """Main chat logic. Retrieves docs and generates a response."""
    global LAST_RETRIEVED_DOCS
    
    if not message: return ""
    if not selected_files: return "Please select at least one file from the left panel."

    # 1. Retrieve
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 7, "filter": {"source": {"$in": selected_files}}}
    )
    
    # Capture docs
    docs = retriever.invoke(message)
    LAST_RETRIEVED_DOCS = docs 

    # 2. Generate
    # System prompt
    template = (
        "You are a helpful assistant. Use the following context to answer the user's question.\n"
        "If the answer is not in the context, simply say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm | StrOutputParser()
    
    context_text = "\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({"context": context_text, "question": message})
    
    return response

def visualize_extended_context():
    """Generates HTML showing previous, current, and next chunks."""
    global LAST_RETRIEVED_DOCS, GLOBAL_CHUNK_STORE
    
    if not LAST_RETRIEVED_DOCS:
        return "<div style='padding:20px; text-align:center;'>No context available. Please ask a question in the chat first.</div>"

    html_output = f"<h3 style='margin-bottom:20px; color: #333;'>Context used for the last response</h3>"

    for i, doc in enumerate(LAST_RETRIEVED_DOCS):
        source = doc.metadata.get("source", "Unknown")
        idx = doc.metadata.get("chunk_index", 0)
        
        # Look up neighbors in the global store
        text_prev = GLOBAL_CHUNK_STORE.get((source, idx - 1), "<i>(Start of document)</i>")
        text_curr = doc.page_content
        text_next = GLOBAL_CHUNK_STORE.get((source, idx + 1), "<i>(End of document)</i>")

        # --- HTML generation ---
        html_output += f"""
        <div style="border: 1px solid #ccc; border-radius: 8px; margin-bottom: 30px; overflow: hidden; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background-color: white;">
            
            <div style="background-color: #e0e0e0; padding: 8px 15px; border-bottom: 1px solid #999; font-weight: bold; font-size: 0.9em; color: #000000;">
                Result #{i+1} | File: {source} | Chunk ID: {idx}
            </div>
            
            <div style="background-color: #fff3e0; padding: 10px; font-size: 0.85em; color: #444444; border-bottom: 1px dotted #ccc;">
                <strong style="color: #d84315;">Previous context:</strong><br>
                {text_prev}
            </div>

            <div style="background-color: #f1f8e9; padding: 15px; font-size: 1em; border-left: 5px solid #4caf50; color: #000000;">
                <strong style="color: #2e7d32;">Retrieved chunk:</strong><br>
                {text_curr}
            </div>

            <div style="background-color: #e3f2fd; padding: 10px; font-size: 0.85em; color: #444444; border-top: 1px dotted #ccc;">
                <strong style="color: #1565c0;">Next context:</strong><br>
                {text_next}
            </div>
        </div>
        """
    
    return html_output

# --- 3. Gradio interface ---

with gr.Blocks(title="RAG") as demo:
    gr.Markdown("# Basic RAG")
    
    with gr.Row():
        # --- Left column: file managment ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Documents")
            file_upload = gr.File(file_count="multiple", label="Upload files")
            upload_btn = gr.Button("Process files", variant="primary")
            status_msg = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### 2. Active context")
            file_selector = gr.CheckboxGroup(label="Available files", choices=[])

        # --- Right column: chat and inspector ---
        with gr.Column(scale=3):
            with gr.Tabs():
                
                # Tab 1: chat
                with gr.TabItem("Chatbot"):
                    chatbot = gr.ChatInterface(
                        fn=chat_response,
                        additional_inputs=[file_selector],
                        description="Ask questions based on the selected documents."
                    )

                # Tab 2: inspector
                with gr.TabItem("Inspector"):
                    gr.Markdown("Review the exact evidence used for the **last answer**.")
                    
                    btn_refresh = gr.Button("Load context", variant="secondary")
                    
                    # HTML component
                    html_viewer = gr.HTML(label="Chunk visualization")

                    btn_refresh.click(
                        fn=visualize_extended_context,
                        inputs=[],
                        outputs=[html_viewer]
                    )

    # --- Event listeners ---
    upload_btn.click(
        fn=process_files,
        inputs=[file_upload],
        outputs=[file_selector, status_msg]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())