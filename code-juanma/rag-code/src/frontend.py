import gradio as gr
import requests
import os
import uuid

# API_URL = "http://127.0.0.1:8001"
API_URL = "http://backend:8001"

# Internal cache to store the hierarchy from the backend
DB_CACHE = {"hierarchy": {}}

# --- API communication functions ---

def load_existing_files_ui():
    """Initial load of the database hierarchy."""
    global DB_CACHE
    print("Requesting hierarchy from the backend...")
    try:
        response = requests.get(f"{API_URL}/files")
        response.raise_for_status()
        DB_CACHE["hierarchy"] = response.json().get("hierarchy", {})
        
        courses = sorted(list(DB_CACHE["hierarchy"].keys()))
        
        all_files = []
        for course, degrees in DB_CACHE["hierarchy"].items():
            for degree, files in degrees.items():
                for file_display in files:
                    clean_filename = file_display.split("] ", 1)[-1]
                    all_files.append(f"[{course}] {degree} - {clean_filename}")
        
        return (
            gr.update(choices=courses, value=[]), 
            gr.update(choices=[], value=[]),
            gr.update(choices=sorted(all_files), value=None)
        )
    except Exception as e:
        print(f"Error loading hierarchy: {e}")
        return gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), gr.update(choices=[], value=None)


def update_degree_dropdown(selected_courses):
    """Updates the degree list based on selected course(s)."""
    degrees = set()
    for course in selected_courses:
        if course in DB_CACHE["hierarchy"]:
            for degree in DB_CACHE["hierarchy"][course].keys():
                degrees.add(f"{course} - {degree}")
    
    return gr.update(choices=sorted(list(degrees)), value=[])

def process_files_ui(files):
    """Uploads new files to the backend."""
    if not files:
        return gr.update(), gr.update(), "No files were uploaded."

    print(f"Uploading {len(files)} files to the backend...")
    upload_data = [('files', (os.path.basename(f.name), open(f.name, 'rb'))) for f in files]
    
    try:
        response = requests.post(f"{API_URL}/upload", files=upload_data)
        response.raise_for_status()
        
        # Refresh the whole UI after upload
        return load_existing_files_ui() + ("Upload completed successfully.",)
    except Exception as e:
        print(f"Error uploading files: {e}")
        return gr.update(), gr.update(), f"Error connecting to backend: {e}"

def delete_file_ui(selected_file_to_delete):
    """Deletes a file from the database and MinIO"""
    if not selected_file_to_delete:
        return gr.update(), gr.update(), gr.update(), "Please select a file to delete."

    try:
        parts = selected_file_to_delete.replace("[", "").split("] ", 1)
        course_part = parts[0]
        
        degree_and_file = parts[1].split(" - ", 1)
        if len(degree_and_file) == 2:
             degree_part = degree_and_file[0]
             filename_part = degree_and_file[1]
        else:
             degree_part = "Unknown"
             filename_part = parts[1]

        payload = {
            "filename": filename_part,
            "course": course_part,
            "degree": degree_part
        }
        
        response = requests.post(f"{API_URL}/delete_file", data=payload)
        response.raise_for_status()
        
        return load_existing_files_ui() + (f"File deleted successfully: {filename_part}",)
        
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), f"Error connecting to backend: {e}"

def chat_response_ui(message, history, selected_courses, selected_degrees, session_id):
    """Bridge between ChatInterface and RAG backend."""

    if not message: 
        return ""
    if not selected_degrees: 
        return "Please select at least one Course and Degree from the left panel."

    selected_context = []
    
    for course in selected_courses:
        if course in DB_CACHE["hierarchy"]:
            for degree_selection in selected_degrees:
                actual_degree = degree_selection.split(" - ", 1)[-1]
                if degree_selection.startswith(f"{course} -"):
                    if actual_degree in DB_CACHE["hierarchy"][course]:
                        for display_name in DB_CACHE["hierarchy"][course][actual_degree]:
                            raw_filename = display_name.split("] ", 1)[-1]
                            selected_context.append({
                                "course": course,
                                "degree": actual_degree,
                                "source": raw_filename
                            })
    
    payload = {
        "message": message, 
        "selected_context": selected_context,
        "chat_history": history,
        "session_id": session_id
    }
    
    try:
        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Error reading response from backend.")
    except Exception as e:
        return f"Error connecting to backend: {e}"

def visualize_extended_context_ui(session_id):
    """Fetches the HTML visualization of the chunks used."""
    try:
        response = requests.get(f"{API_URL}/inspector?session_id={session_id}")
        response.raise_for_status()
        return response.json().get("html", "")
    except Exception as e:
        return f"<p>Error connecting to backend: {e}</p>"


# --- Gradio interface ---

with gr.Blocks(title="RAG DIA") as demo:
    gr.Markdown("# RAG DIA")
    session_state = gr.State(lambda: str(uuid.uuid4()))

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Documents")
            file_upload = gr.File(file_count="multiple", label="Upload new files")
            upload_btn = gr.Button("Process files", variant="primary")
            status_msg = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Delete files")
            file_to_delete_dropdown = gr.Dropdown(label="Select file to delete", choices=[])
            delete_btn = gr.Button("Delete selected file", variant="stop")
            delete_status_msg = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### 2. Context active")
            btn_refresh_files = gr.Button("🔄 Refresh list of DB", variant="secondary", size="sm")
            
            course_selector = gr.CheckboxGroup(
                label="Select Course(s)", 
                choices=[]
            )
            
            degree_selector = gr.CheckboxGroup(
                label="Select Degree/Master", 
                choices=[]
            )

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Chatbot"):
                    chatbot = gr.ChatInterface(
                        fn=chat_response_ui,
                        additional_inputs=[course_selector, degree_selector, session_state],
                        description="Ask questions. The system will search in the selected documents."
                    )

                with gr.TabItem("Inspector"):
                    gr.Markdown("Review the exact fragments used for the last response.")
                    btn_refresh_context = gr.Button("Load context of the last response", variant="secondary")
                    html_viewer = gr.HTML(label="Chunks visualization")

                    btn_refresh_context.click(
                        fn=visualize_extended_context_ui,
                        inputs=[session_state],
                        outputs=[html_viewer]
                    )

    # --- Events ---
    
    # Load data on page load
    demo.load(
        fn=load_existing_files_ui,
        inputs=[],
        outputs=[course_selector, degree_selector, file_to_delete_dropdown]
    )

    # Manual refresh
    btn_refresh_files.click(
        fn=load_existing_files_ui,
        inputs=[],
        outputs=[course_selector, degree_selector, file_to_delete_dropdown]
    )

    # Cascaded update: When course changes, update degrees
    course_selector.change(
        fn=update_degree_dropdown,
        inputs=[course_selector],
        outputs=[degree_selector]
    )

    # Upload and refresh
    upload_btn.click(
        fn=process_files_ui,
        inputs=[file_upload],
        outputs=[course_selector, degree_selector, file_to_delete_dropdown,status_msg]
    )

    # Delete file event
    delete_btn.click(
        fn=delete_file_ui,
        inputs=[file_to_delete_dropdown],
        outputs=[course_selector, degree_selector, file_to_delete_dropdown, delete_status_msg]
    )

if __name__ == "__main__":
    demo.launch(
        server_port=7860, 
        server_name="0.0.0.0",
        theme=gr.themes.Soft()
    )