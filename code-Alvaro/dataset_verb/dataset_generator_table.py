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
# 2. LLM CONFIGURATION
# ==========================================

llm_extractor = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.84.51.82:5000/v1",
    api_key="not_required",
    temperature=0.1
)

llm_generator = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.84.51.82:5000/v1",
    api_key="not_required",
    temperature=0.7
)

# ==========================================
# 3. PIPELINE FUNCTIONS
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


def extract_tables_as_text(markdown_text: str) -> str:
    """Extrae solo las tablas del markdown como texto para el generador."""
    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=ExportType.MARKDOWN
    )
    docs = loader.load()
    markdown_text = "\n".join([doc.page_content for doc in docs])
    
    tables = extract_markdown_tables(markdown_text)
    if not tables:
        return ""
    
    tables_text = ""
    for i, t in enumerate(tables):
        tables_text += f"\n--- TABLA {i+1} ---\n"
        tables_text += f"Contexto: {t['context']}\n"
        tables_text += f"{t['table']}\n"
    
    return tables_text


def extract_schema(guide_text: str) -> CourseGuideSchema:
    """Extracts structured facts from the Markdown text."""
    print("-> Extracting schema from the course guide...")

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
    - Para preguntas Factual simples (créditos, profesor, fecha): 1-2 frases con el dato exacto.
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


def generate_table_questions(tables_text: str, course_name: str, num_questions: int = 15) -> QADataset:
    print(f"-> Generating {num_questions} table-specific questions...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un simulador avanzado de estudiantes universitarios.

Genera {num_questions} pares de pregunta y respuesta basados EXCLUSIVAMENTE en las tablas de la asignatura "{course_name}".

REQUISITO DE IDIOMA: Todo en español.

REGLAS PARA LAS PREGUNTAS:
- Cada pregunta DEBE mencionar explícitamente "{course_name}".
- Las preguntas deben requerir leer una tabla para ser respondidas.
- Ejemplos de buenas preguntas sobre tablas:
  "¿En qué semana se realiza el test del Tema 1 en {course_name}?"
  "¿Qué peso tiene la primera práctica en grupo en {course_name}?"
  "¿Cuál es la nota mínima para aprobar el test de {course_name}?"

REGLAS PARA ground_truth:
- Debe ser COMPLETO y AUTOCONTENIDO con todos los datos de la tabla.
- Incluye SIEMPRE los valores numéricos exactos: semanas, porcentajes, notas mínimas.
- NUNCA omitas datos con "entre otros" o "etc."

REGLAS PARA ground_truth_context:
- Copia literalmente las celdas relevantes de la tabla original.
- Mantén la correspondencia fila-columna exacta.

TAXONOMÍA (solo estas):
1. Factual — dato concreto de una celda
2. Summarization — resumen de toda una tabla
3. Multi-hop Reasoning — combina datos de varias filas o tablas

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
        ("human", "ASIGNATURA: {course_name}\n\nTABLAS DEL DOCUMENTO:\n{tables_text}\n\nResponde solo con JSON:")
    ])

    chain = prompt | llm_generator
    result = chain.invoke({
        "num_questions": num_questions,
        "course_name": course_name,
        "tables_text": tables_text
    })
    raw = result.content if hasattr(result, "content") else str(result)
    parsed = _parse_json_response(raw)
    return QADataset(**parsed)

def clean_course_name(raw: str) -> str:
    # Elimina patrones tipo "105000008 - " o "105000008: "
    cleaned = re.sub(r"^\d+\s*[-:]\s*", "", raw.strip())
    return cleaned

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    pdf_path = "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/"
    NUM_QUESTIONS = 20

    # A. Cargar en DOC_CHUNKS para el schema general
    loader_chunks = DoclingLoader(file_path=pdf_path, export_type=ExportType.DOC_CHUNKS)
    docs = loader_chunks.load()
    full_text = "\n".join([doc.page_content for doc in docs])

    # B. Cargar en MARKDOWN para extraer tablas
    loader_md = DoclingLoader(file_path=pdf_path, export_type=ExportType.MARKDOWN)
    docs_md = loader_md.load()
    markdown_text = "\n".join([doc.page_content for doc in docs_md])

    # C. Extraer tablas
    tables_text = extract_tables_as_text(markdown_text)
    print(f"\n--- TABLAS ENCONTRADAS ---\n{tables_text[:1000]}")

    if not tables_text:
        print("⚠️ No se encontraron tablas en el documento.")
    else:
        # D. Extraer schema para el course_name
        extracted_schema = extract_schema(full_text)
        course_name = clean_course_name(extracted_schema.course_name)

        # E. Generar preguntas exclusivamente sobre tablas
        final_dataset = generate_table_questions(tables_text, course_name=course_name, num_questions=NUM_QUESTIONS)

        pdf_filename = Path(pdf_path).name
        for pair in final_dataset.questions:
            pair.source_document = pdf_filename

        # F. Guardar
        output_file = "dataset_tables.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_dataset.model_dump(), f, ensure_ascii=False, indent=4)

        print(f"\nDone! Generated {len(final_dataset.questions)} table questions → '{output_file}'")