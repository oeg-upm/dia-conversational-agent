"""
frontend/app.py
---------------
Gradio frontend for the RAG DIA backend.
Selector jerárquico en 3 niveles: Curso → Titulación → Documento
"""

import os
import requests
import gradio as gr

API_URL = os.environ.get("API_URL", "http://localhost:9000")

# Caché local de la jerarquía devuelta por el backend
DB_CACHE = {"hierarchy": {}}


# ---------------------------------------------------------------------------
# Funciones de comunicación con la API
# ---------------------------------------------------------------------------

def load_existing_files_ui():
    """Carga inicial (y refresco manual) de la jerarquía del backend."""
    global DB_CACHE
    print("Requesting hierarchy from backend...")
    try:
        response = requests.get(f"{API_URL}/list_documents", timeout=10)
        response.raise_for_status()
        DB_CACHE["hierarchy"] = response.json().get("hierarchy", {})

        courses = sorted(list(DB_CACHE["hierarchy"].keys()))
        print(f"Courses loaded: {courses}")

        # Al refrescar, reseteamos los tres selectores
        return (
            gr.update(choices=courses, value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
        )
    except Exception as e:
        print(f"Error loading hierarchy: {e}")
        return (
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
        )


def update_degree_dropdown(selected_courses):
    """Nivel 2: actualiza titulaciones según los cursos seleccionados."""
    degrees = set()
    for course in selected_courses:
        if course in DB_CACHE["hierarchy"]:
            for degree in DB_CACHE["hierarchy"][course].keys():
                degrees.add(f"{course} - {degree}")

    # Al cambiar curso, reseteamos titulación y documentos
    return (
        gr.update(choices=sorted(list(degrees)), value=[]),
        gr.update(choices=[], value=[]),
    )


def update_document_dropdown(selected_degrees):
    """Nivel 3: actualiza documentos según las titulaciones seleccionadas."""
    docs = []
    for entry in selected_degrees:
        # entry tiene el formato "curso - titulación"
        if " - " not in entry:
            continue
        course, degree_key = entry.split(" - ", 1)
        if course not in DB_CACHE["hierarchy"]:
            continue
        if degree_key not in DB_CACHE["hierarchy"][course]:
            continue
        for display_name in DB_CACHE["hierarchy"][course][degree_key]:
            # display_name tiene el formato "[curso] filename"
            raw_filename = display_name.split("] ", 1)[-1]
            label = f"{course} | {degree_key} | {raw_filename}"
            if label not in docs:
                docs.append(label)

    return gr.update(choices=sorted(docs), value=sorted(docs))


def process_files_ui(files, course, category, degree):
    """Sube archivos al backend con sus metadatos jerárquicos."""
    if not files:
        return gr.update(), gr.update(), gr.update(), "No files were uploaded."

    print(f"Uploading {len(files)} file(s) to backend...")

    handles = []
    upload_data = []
    for f in files:
        fh = open(f.name, "rb")
        handles.append(fh)
        upload_data.append(("files", (os.path.basename(f.name), fh, "application/pdf")))

    try:
        response = requests.post(
            f"{API_URL}/upload",
            data={"course": course, "category": category, "degree": degree},
            files=upload_data,
            timeout=600,
        )
        response.raise_for_status()
        msg = response.json().get("status_message", "Upload completed successfully.")
    except Exception as e:
        print(f"Error uploading files: {e}")
        msg = f"Error connecting to backend: {e}"
    finally:
        for fh in handles:
            fh.close()

    course_upd, degree_upd, doc_upd = load_existing_files_ui()
    return course_upd, degree_upd, doc_upd, msg


def chat_response_ui(message, history, selected_docs):
    """Construye el contexto a partir de los documentos seleccionados y llama a /chat."""
    if not message:
        return ""
    if not selected_docs:
        return "Por favor, selecciona al menos un documento en el panel izquierdo."

    selected_context = []
    for label in selected_docs:
        # label tiene el formato "curso | titulación | filename"
        parts = label.split(" | ", 2)
        if len(parts) != 3:
            continue
        course, degree_key, raw_filename = parts
        selected_context.append({
            "course": course,
            "degree": degree_key,
            "source": raw_filename,
        })

    if not selected_context:
        return "No se encontraron documentos para la selección actual. Prueba a refrescar la lista."

    payload = {
        "message": message,
        "selected_context": selected_context,
        "chat_history": history,
    }

    try:
        response = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", data.get("answer", "Error leyendo la respuesta."))
    except Exception as e:
        return f"Error conectando con el backend: {e}"


def visualize_extended_context_ui():
    """Obtiene el HTML del inspector de chunks para la última consulta."""
    try:
        response = requests.get(f"{API_URL}/inspector", timeout=10)
        response.raise_for_status()
        return response.json().get("html", "<p>No hay contexto disponible todavía.</p>")
    except Exception as e:
        return f"<p>Error conectando con el backend: {e}</p>"


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAG DIA") as demo:

    gr.Markdown("# RAG DIA")

    with gr.Row():

        # ------------------------------------------------------------------
        # Columna izquierda: upload + selector jerárquico en 3 niveles
        # ------------------------------------------------------------------
        with gr.Column(scale=1):

            gr.Markdown("### 1. Subir documentos")

            with gr.Group():
                course_input   = gr.Textbox(label="Curso / Año",  placeholder="ej. 2024")
                category_input = gr.Textbox(label="Categoría",    placeholder="ej. Grado / Máster")
                degree_input   = gr.Textbox(label="Titulación",   placeholder="ej. Ingeniería Informática")

            file_upload = gr.File(file_count="multiple", label="Archivos PDF")
            upload_btn  = gr.Button("Procesar archivos", variant="primary")
            status_msg  = gr.Textbox(label="Estado", interactive=False)

            gr.Markdown("### 2. Seleccionar contexto")

            btn_refresh_files = gr.Button("🔄 Refrescar lista de BD", variant="secondary", size="sm")

            course_selector = gr.CheckboxGroup(
                label="Curso(s)",
                choices=[],
            )
            degree_selector = gr.CheckboxGroup(
                label="Titulación / Máster",
                choices=[],
            )
            doc_selector = gr.CheckboxGroup(
                label="Documentos",
                choices=[],
            )

        # ------------------------------------------------------------------
        # Columna derecha: chatbot e inspector
        # ------------------------------------------------------------------
        with gr.Column(scale=3):

            with gr.Tabs():

                with gr.TabItem("Chatbot"):
                    chatbot = gr.ChatInterface(
                        fn=chat_response_ui,
                        additional_inputs=[doc_selector],
                        description="Haz preguntas. El sistema buscará en los documentos seleccionados.",
                    )

                with gr.TabItem("Inspector"):
                    gr.Markdown("Revisa los fragmentos exactos usados para la última respuesta.")
                    btn_refresh_context = gr.Button(
                        "Cargar contexto de la última respuesta", variant="secondary"
                    )
                    html_viewer = gr.HTML(label="Visualización de chunks")

                    btn_refresh_context.click(
                        fn=visualize_extended_context_ui,
                        inputs=[],
                        outputs=[html_viewer],
                    )

    # -----------------------------------------------------------------------
    # Eventos
    # -----------------------------------------------------------------------

    demo.load(
        fn=load_existing_files_ui,
        inputs=[],
        outputs=[course_selector, degree_selector, doc_selector],
    )

    btn_refresh_files.click(
        fn=load_existing_files_ui,
        inputs=[],
        outputs=[course_selector, degree_selector, doc_selector],
    )

    # Cascada nivel 1 → 2: curso cambia → actualiza titulaciones y resetea docs
    course_selector.change(
        fn=update_degree_dropdown,
        inputs=[course_selector],
        outputs=[degree_selector, doc_selector],
    )

    # Cascada nivel 2 → 3: titulación cambia → actualiza documentos
    degree_selector.change(
        fn=update_document_dropdown,
        inputs=[degree_selector],
        outputs=[doc_selector],
    )

    # Upload → refresca los tres niveles
    upload_btn.click(
        fn=process_files_ui,
        inputs=[file_upload, course_input, category_input, degree_input],
        outputs=[course_selector, degree_selector, doc_selector, status_msg],
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        server_name="0.0.0.0",
        theme=gr.themes.Soft(),
    )