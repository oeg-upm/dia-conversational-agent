import os
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

import pypandoc

# ==========================================
# CONFIG
# ==========================================

INPUT_BASE_DIR = "/home/alvaro/Escritorio/Guías aprendizaje"
OUTPUT_BASE_DIR = "/home/alvaro/Escritorio/Guias_verbalizadas"

# ==========================================
# LLM
# ==========================================

llm_verbalizer = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.92.139.47:5000/v1",
    api_key="not_required",
    temperature=0.2,
    timeout=300  # ⏱️ reducido
)

# ==========================================
# PDF → MARKDOWN (OCR OFF)
# ==========================================

def load_pdf_as_markdown(pdf_path: str) -> str:
    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=ExportType.MARKDOWN
    )
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

# ==========================================
# EXTRACCIÓN
# ==========================================

def extract_markdown_tables(markdown_text: str) -> List[dict]:
    lines = markdown_text.split("\n")
    tables = []
    i = 0

    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and re.match(r"[\|\s\-:]+", lines[i + 1]):
            start = i
            while i < len(lines) and "|" in lines[i]:
                i += 1
            end = i

            context = "\n".join(lines[max(0, start - 2):start])

            tables.append({
                "table": "\n".join(lines[start:end]),
                "context": context,
                "start": start,
                "end": end
            })
        else:
            i += 1

    return tables


def extract_image_references(markdown_text: str) -> List[dict]:
    images = []

    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]*)\)", markdown_text):
        images.append({
            "reference": match.group(0),
            "caption": match.group(1) or "Sin descripción"
        })

    return images

# ==========================================
# MARKDOWN → PDF
# ==========================================

def save_markdown_as_pdf(markdown_text: str, output_path: str):
    title = os.path.basename(output_path).replace(".pdf", "")

    pypandoc.convert_text(
        markdown_text,
        to="pdf",
        format="markdown",
        outputfile=output_path,
        extra_args=[
            "--standalone",
            "--metadata", f"title={title}",  # 🧠 título automático
            "--pdf-engine=wkhtmltopdf"
        ]
    )

# ==========================================
# VERBALIZACIÓN
# ==========================================

def verbalize_table(table_markdown: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages([("system", """Eres un asistente académico experto en guías docentes universitarias.
Tu tarea es convertir una tabla en formato Markdown en texto en español.

OBJETIVO PRINCIPAL: mantener la fidelidad exacta de la información.

REGLAS:
1. Cada fila de la tabla debe ser representada explícitamente en el texto.
2. Mantén la correspondencia exacta entre columnas y valores.
3. NO agrupes filas si eso puede generar ambigüedad.
4. Conserva TODOS los datos: nombres, porcentajes, fechas, pesos, condiciones mínimas, etc.
5. Usa frases claras y completas, pero prioriza la precisión sobre la naturalidad.
6. NO inventes datos.
7. Output únicamente el texto final."""),
        ("human", "CONTEXTO (título o sección):\n{context}\n\nTABLA:\n{table}\n\nConvierte en prosa narrativa:")
    ])

    chain = prompt | llm_verbalizer
    return chain.invoke({"table": table_markdown, "context": context}).content


def verbalize_image(caption: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente académico experto en guías docentes universitarias.
Una guía docente contiene una imagen o figura. Basándote ÚNICAMENTE en su pie de foto y el texto cercano,
escribe una breve descripción en español de lo que probablemente representa esa imagen.

REGLAS:
1. Indica explícitamente que es una figura o imagen del documento.
2. Si no hay suficiente contexto, escribe: "El documento incluye una figura sin descripción detallada en esta sección."
3. No inventes contenido que no puedas inferir del contexto.
4. Output ÚNICAMENTE la descripción, sin explicaciones ni comentarios."""),
        ("human", "PIE DE FOTO: {caption}\n\nCONTEXTO CERCANO:\n{surrounding_text}\n\nDescribe esta imagen:")
    ])

    chain = prompt | llm_verbalizer
    return chain.invoke({"caption": caption, "context": context}).content


def verbalize_markdown(markdown_text: str) -> str:
    lines = markdown_text.split("\n")
    result = lines.copy()

    # TABLAS
    tables = extract_markdown_tables(markdown_text)

    for table in reversed(tables):
        text = verbalize_table(table["table"], table["context"])
        result[table["start"]:table["end"]] = [f"\n{text}\n"]

    text = "\n".join(result)

    # IMÁGENES
    images = extract_image_references(text)

    for img in images:
        pos = text.find(img["reference"])
        context = text[max(0, pos - 200):pos + 200]

        desc = verbalize_image(img["caption"], context)

        text = text.replace(img["reference"], f"\n[FIGURA: {desc}]\n", 1)

    return text

# ==========================================
# PIPELINE
# ==========================================

def process_pdf(pdf_path: str):
    try:
        print(f"Processing: {pdf_path}")

        relative = os.path.relpath(pdf_path, INPUT_BASE_DIR)
        output_path = os.path.join(OUTPUT_BASE_DIR, relative)
        output_path = os.path.splitext(output_path)[0] + ".pdf"

        if os.path.exists(output_path):
            print("   Skipped (already processed)")
            return

        markdown = load_pdf_as_markdown(pdf_path)
        verbalized = verbalize_markdown(markdown)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        save_markdown_as_pdf(verbalized, output_path)

        print(f"   Saved: {output_path}")

    except Exception as e:
        print(f"\n❌ Error en: {pdf_path}")
        print(f"Detalle: {e}")

# ==========================================
# WALKER
# ==========================================

def walk_pdfs(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                yield os.path.join(root, f)

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    pdfs = list(walk_pdfs(INPUT_BASE_DIR))
    total = len(pdfs)

    print(f"Found {total} PDFs\n")

    max_workers = 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdfs}

        for i, future in enumerate(as_completed(futures), 1):
            pdf = futures[future]
            try:
                future.result()
                print(f"[{i}/{total}] Done → {os.path.basename(pdf)}")
            except Exception as e:
                print(f"[{i}/{total}] Error → {pdf}: {e}")