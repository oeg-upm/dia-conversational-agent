import json
import re
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from pathlib import Path

# ==========================================
# 1. DATA STRUCTURES DEFINITION (PYDANTIC)
# ==========================================

class CourseGuideSchema(BaseModel):
    course_name: str = Field(
        description="Official name of the course/subject exactly as it appears in the guide (e.g., 'Sistemas de Planificación')."
    )
    course_description: str = Field(
        description="Brief summary or general description of what the course is about and its main goal."
    )
    general_info: str = Field(
        description="Course name, credits (ECTS), semester, course type (optativa/obligatoria), academic year, language, etc."
    )
    teaching_staff: str = Field(
        description="Names, roles (coordinator/professor), emails, office room numbers, and tutoring hours."
    )
    prerequisites: str = Field(
        description="Previous knowledge, recommended prior courses, or mandatory prerequisites."
    )
    competencies_and_outcomes: str = Field(
        description="General and specific competencies to be acquired, and expected learning outcomes (RA)."
    )
    syllabus: str = Field(
        description="Breakdown of the course topics, modules, or units (Temario)."
    )
    schedule: str = Field(
        description="Chronological schedule (cronograma), week-by-week activities, or important dates for theory classes, labs, and evaluations."
    )
    evaluation_criteria: str = Field(
        description="Detailed criteria for continuous evaluation and final exams, including weights/percentages, minimum grades, and types of tests."
    )
    bibliography_and_resources: str = Field(
        description="Required and recommended books, software, websites, or other learning resources."
    )

class QAPair(BaseModel):
    question: str = Field(description="The generated question simulating a student's query (in Spanish).")
    ground_truth: str = Field(description="The ideal, factual answer based strictly on the course guide (in Spanish).")
    ground_truth_context: str = Field(description="The exact extract from the course guide that supports the ground_truth answer (in Spanish).")
    question_type: str = Field(description="Category: 'Factual', 'Summarization', 'Multi-hop Reasoning', or 'Unanswerable'.")
    student_profile: str = Field(description="Simulated profile: 'Freshman', 'Senior', 'Formal tone', 'Informal tone', etc.")
    source_document: str = Field(default="", description="Filename of the source PDF.")

class QADataset(BaseModel):
    questions: List[QAPair]

# ==========================================
# 2. LOCAL LLM CONFIGURATION
# ==========================================

llm_extractor = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.84.51.82:5000/v1",
    api_key="not_required",
    temperature=0.1
)

llm_verbalizer = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.84.51.82:5000/v1",
    api_key="not_required",
    temperature=0.2
)

llm_generator = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.84.51.82:5000/v1",
    api_key="not_required",
    temperature=0.7,
    max_tokens=8192
)

# ==========================================
# 3. PDF LOADING — DOCLINGLOADER MARKDOWN EXPORT
# ==========================================

def load_pdf_as_markdown(pdf_path: str) -> str:
    """
    Loads the PDF using DoclingLoader with ExportType.MARKDOWN.
    Docling runs its layout detection model (RT-DETRv2) to identify tables,
    figures, headings, and body text as separate structural elements.
    The Markdown export preserves this structure:
      - Tables  → proper Markdown table syntax (| col | col |)
      - Figures → image placeholders with captions
      - Headings → # / ## / ### hierarchy
      - Body text → clean paragraphs
    """
    print(f"-> Loading PDF and exporting to Markdown via DoclingLoader: {pdf_path}")

    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=ExportType.MARKDOWN
    )
    docs = loader.load()
    markdown_text = "\n".join([doc.page_content for doc in docs])

    print(f"   Markdown export complete ({len(markdown_text)} chars).")
    return markdown_text


# ==========================================
# 4. SELECTIVE VERBALIZATION
# ==========================================

def extract_markdown_tables(markdown_text: str) -> List[dict]:
    """
    Detects all Markdown tables in the text and returns them with their
    position and surrounding context (heading/caption lines just before
    the table, used to give the verbalizer semantic context).
    """
    lines = markdown_text.split("\n")
    tables = []
    i = 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and re.match(r"[\|\s\-:]+", lines[i + 1]):
            table_start = i
            while i < len(lines) and "|" in lines[i]:
                i += 1
            table_end = i
            context_start = max(0, table_start - 2)
            context = "\n".join(lines[context_start:table_start])
            tables.append({
                "table": "\n".join(lines[table_start:table_end]),
                "context": context,
                "start": table_start,
                "end": table_end
            })
        else:
            i += 1
    return tables


def extract_image_references(markdown_text: str) -> List[dict]:
    """
    Finds all image/figure references exported by Docling:
      - Standard Markdown images: ![caption](path)
      - Docling HTML-style comments: <!-- image --> or <!-- image: caption -->
    """
    images = []
    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]*)\)", markdown_text):
        images.append({"reference": match.group(0), "caption": match.group(1) or "Sin descripción"})
    for match in re.finditer(r"<!-- image(?::([^>]*))? -->", markdown_text):
        caption = match.group(1) or "figura sin descripción"
        images.append({"reference": match.group(0), "caption": caption})
    return images


def verbalize_table(table_markdown: str, context: str) -> str:
    """
    Converts a single Markdown table into natural language prose using the LLM.
    Preserves all factual data (percentages, names, dates, conditions, etc.)
    while making it readable as flowing text.
    """
    prompt = ChatPromptTemplate.from_messages([
#         ("system", """Eres un asistente académico experto en guías docentes universitarias.
# Tu tarea es convertir una tabla en formato Markdown en una descripción narrativa en prosa fluida en español.

# REGLAS:
# 1. Conserva TODOS los datos: nombres, porcentajes, fechas, pesos, condiciones mínimas, etc.
# 2. Escribe frases completas y naturales. Por ejemplo: "La evaluación continua tiene un peso del 60% de la nota final."
# 3. Si la tabla tiene muchas filas, agrúpalas semánticamente en lugar de listarlas una a una.
# 4. NO inventes datos que no estén en la tabla.
# 5. Output ÚNICAMENTE el texto narrativo, sin explicaciones ni comentarios."""),
("system", """Eres un asistente académico experto en guías docentes universitarias.
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
    result = chain.invoke({"table": table_markdown, "context": context})
    return result.content


def verbalize_image(image_reference: str, caption: str, surrounding_text: str) -> str:
    """
    Generates a descriptive sentence for a figure/image based on its caption
    and the text surrounding it in the document.
    """
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
    result = chain.invoke({"caption": caption, "surrounding_text": surrounding_text[:500]})
    return result.content


def verbalize_markdown(markdown_text: str) -> str:
    """
    Main verbalization function. Selectively verbalizes only tables and images.
    Plain text, headings and paragraphs are kept exactly as-is.
    """
    print("-> Verbalizing tables and images (plain text kept as-is)...")

    lines = markdown_text.split("\n")
    result_lines = lines.copy()

    # --- Verbalize tables (bottom-to-top to preserve line indices) ---
    tables = extract_markdown_tables(markdown_text)
    print(f"   Found {len(tables)} table(s).")
    for i, table_info in enumerate(reversed(tables)):
        print(f"   Verbalizing table {len(tables) - i}/{len(tables)}...")
        verbalized = verbalize_table(table_info["table"], table_info["context"])
        result_lines[table_info["start"]:table_info["end"]] = [f"\n{verbalized}\n"]

    verbalized_text = "\n".join(result_lines)

    # --- Describe image references ---
    images = extract_image_references(verbalized_text)
    print(f"   Found {len(images)} image reference(s).")
    for i, img in enumerate(images):
        print(f"   Describing image {i + 1}/{len(images)}...")
        pos = verbalized_text.find(img["reference"])
        surrounding = verbalized_text[max(0, pos - 200): pos + 200]
        description = verbalize_image(img["reference"], img["caption"], surrounding)
        verbalized_text = verbalized_text.replace(img["reference"], f"\n[FIGURA: {description}]\n", 1)

    print("   Verbalization complete.")
    return verbalized_text


# ==========================================
# 5. SCHEMA EXTRACTION & Q&A GENERATION
# ==========================================

def _parse_json_response(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")


def extract_schema(guide_text: str) -> CourseGuideSchema:
    """Extracts structured facts from the verbalized document."""
    print("-> Extracting schema from the verbalized course guide...")

    schema_fields = "\n".join([
        f'  "{name}": "<{field.description}>"'
        for name, field in CourseGuideSchema.model_fields.items()
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Eres un asistente académico experto. Lee la guía docente universitaria y extrae la información clave.
Ignora el relleno y conserva solo los hechos puros. Escribe tu output en español.
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin comillas de código markdown.

El JSON debe tener exactamente estas claves:
{{{{
{schema_fields}
}}}}"""),
        ("human", "Aquí está el texto de la guía docente:\n\n{text}\n\nResponde solo con el JSON:")
    ])

    chain = prompt | llm_extractor
    result = chain.invoke({"text": guide_text})
    raw = result.content if hasattr(result, "content") else str(result)

    parsed = _parse_json_response(raw)
    return CourseGuideSchema(**parsed)


def generate_questions(schema: CourseGuideSchema, course_name: str, num_questions: int = 20) -> QADataset:
    """Generates diverse Q&A pairs based on the extracted schema."""
    print(f"-> Generating a diverse dataset of {num_questions} questions...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un simulador avanzado de estudiantes universitarios.

    Genera {num_questions} pares de pregunta y respuesta basados en los datos de la asignatura "{course_name}".

    REQUISITO DE IDIOMA: Todo en español.

    REGLAS PARA LAS PREGUNTAS:
    - Cada pregunta DEBE mencionar explícitamente el nombre de la asignatura "{course_name}".
        Ejemplo correcto: "¿Cuál es la bibliografía básica de {course_name}?"
        Ejemplo incorrecto: "¿Cuál es la bibliografía básica del curso?"

    REGLAS PARA ground_truth (MUY IMPORTANTE):
    - Debe ser una respuesta COMPLETA y AUTOCONTENIDA: alguien que solo lea el ground_truth debe entender la respuesta sin necesitar contexto adicional.
    - Para preguntas Summarization y Multi-hop: incluye TODOS los elementos relevantes sin omitir ninguno.
    - Debe ser CONCISA pero INFORMATIVA con los datos exactos (nombres, porcentajes, fechas, créditos).
    - NUNCA uses frases vagas como "se puede encontrar en...", "el curso incluye...", "hay información sobre...".
    - SIEMPRE incluye los datos concretos del schema.
    

        MAL ejemplo: "La bibliografía básica incluye varios libros relevantes para la asignatura."
        BIEN ejemplo: "La bibliografía básica de {course_name} incluye 'Investigación Operativa: Modelos Determinísticos y Estocásticos' de Ríos Insua et al. (2004) y 'Métodos y Modelos de Investigación de Operaciones' de Kaufmann (1972)."

        MAL ejemplo: "La evaluación continua tiene un peso significativo en la nota final."
        BIEN ejemplo: "En {course_name}, la evaluación continua representa el 60% de la nota final, dividida en pruebas parciales (40%) y prácticas de laboratorio (20%). La nota mínima para aprobar es un 4.0."

    REGLAS PARA ground_truth_context:
    - SOLO palabras que aparecen literalmente en el schema proporcionado.
    - NO reformular, NO añadir información.
    - Debe poder encontrarse con CTRL+F en el texto original.


    TAXONOMÍA:
    1. Factual
    2. Summarization
    3. Multi-hop Reasoning
    4. Unanswerable (máximo 3)

    Para preguntas Unanswerable:
    - ground_truth: explicación natural
    - ground_truth_context: "No disponible en el documento"

    FORMATO JSON:
    {{{{
    "questions": [
        {{{{
        "question": "...",
        "ground_truth": "...",
        "ground_truth_context": "...",
        "question_type": "...",
        "student_profile": "..."
        }}}}
    ]
    }}}}"""),

        ("human", "ASIGNATURA: {course_name}\n\nDATOS DEL CURSO:\n{schema_json}\n\nResponde solo con JSON:")
    ])

    schema_json = schema.model_dump_json(indent=2)
    chain = prompt | llm_generator
    result = chain.invoke({
        "num_questions": num_questions,
        "schema_json": schema_json,
        "course_name": course_name
    })
    raw = result.content if hasattr(result, "content") else str(result)

    parsed = _parse_json_response(raw)
    return QADataset(**parsed)



def clean_course_name(raw: str) -> str:
    # Elimina patrones tipo "105000008 - " o "105000008: "
    cleaned = re.sub(r"^\d+\s*[-:]\s*", "", raw.strip())
    return cleaned
# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    pdf_path = "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/"
    NUM_QUESTIONS = 20

    # A. Load PDF → structured Markdown via DoclingLoader
    markdown_text = load_pdf_as_markdown(pdf_path)
    print("\n--- MARKDOWN EXPORT (first 1500 chars) ---")
    print(markdown_text[:1500])

    # B. Save raw Markdown for inspection before any verbalization
    with open("markdown_raw.md", "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print("\n-> Saved raw Docling Markdown to 'markdown_raw.md' (open with VSCode Ctrl+Shift+V to preview)")

    # C. Selectively verbalize tables and images
    verbalized_text = verbalize_markdown(markdown_text)
    print("\n--- VERBALIZED TEXT (first 1500 chars) ---")
    print(verbalized_text[:1500])

    # Save intermediate verbalized text for inspection/debugging
    with open("verbalized_guide.md", "w", encoding="utf-8") as f:
        f.write(verbalized_text)
    print("\n-> Saved verbalized text to 'verbalized_guide.md'")

    # D. Extract structured schema
    extracted_schema = extract_schema(verbalized_text)
    print("\n--- EXTRACTED SCHEMA ---")
    print(extracted_schema.model_dump_json(indent=2))

    # DEBUG: Check if schema fields are empty or generic
    schema_dict = extracted_schema.model_dump()
    empty_fields = [k for k, v in schema_dict.items() if not v or v.strip() == "" or "no se especifica" in v.lower() or "not specified" in v.lower()]
    if empty_fields:
        print(f"\n  WARNING: These schema fields are empty or generic: {empty_fields}")
        print("   The LLM may not have received the document text correctly.")
    else:
        print("\n Schema looks good, all fields populated.")

    # E. Generate Q&A dataset
    #course_name = extracted_schema.course_name  # limpio, sin parsear
    course_name = clean_course_name(extracted_schema.course_name)
    final_dataset = generate_questions(extracted_schema, course_name=course_name, num_questions=NUM_QUESTIONS)
    
    pdf_filename = Path(pdf_path).name  # extrae solo el nombre del archivo PDF

    for pair in final_dataset.questions:
        pair.source_document = pdf_filename

    # DEBUG: Print first 3 questions to check quality before saving
    print("\n--- SAMPLE QUESTIONS (first 3) ---")
    for i, q in enumerate(final_dataset.questions[:3]):
        print(f"\n[{i+1}] Type: {q.question_type} | Profile: {q.student_profile}")
        print(f"     Q: {q.question}")
        print(f"     A: {q.ground_truth[:200]}...")

    # F. Save results
    output_file = "dataset_verbalized.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset.model_dump(), f, ensure_ascii=False, indent=4)

    print(f"\n Done! Generated {len(final_dataset.questions)} questions → '{output_file}'")