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
    base_url="http://100.119.72.127:11434/v1",
    api_key="not_required",
    temperature=0.1
)

llm_generator = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.119.72.127:11434/v1",
    api_key="not_required",
    temperature=0.7
)

# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================

Q_TYPE_DISTRIBUTION = {
    "Factual": 0.45,
    "Summarization": 0.30,
    "Multi-hop Reasoning": 0.15,
    "Unanswerable": 0.10,
}

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
    
    # Calcular distribución
    type_counts = {
        q_type: max(1, round(num_questions * prob))
        for q_type, prob in Q_TYPE_DISTRIBUTION.items()
    }
    
    distribution_str = "\n".join([
        f"    - {q_type}: {count} preguntas"
        for q_type, count in type_counts.items()
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un simulador avanzado de estudiantes universitarios.

    Genera {num_questions} pares de pregunta y respuesta basados en los datos de la asignatura "{course_name}".
    DISTRIBUCIÓN OBLIGATORIA DE PREGUNTAS (debes respetar exactamente estos números):
    {distribution_str}

    REQUISITO DE IDIOMA: Todo en español.

    REGLAS PARA LAS PREGUNTAS:
    - Cada pregunta DEBE mencionar explícitamente el nombre de la asignatura "{course_name}".
        Ejemplo correcto: "¿Cuál es la bibliografía básica de {course_name}?"
        Ejemplo incorrecto: "¿Cuál es la bibliografía básica del curso?"

    Taxonomía de tipo de pregunta:
    1. Factual: Hecho de un solo paso explícitamente indicado en el texto
    2. Summarization: Requiere sintetizar o agrupar múltiples elementos de información.
    3. Multi-hop Reasoning: Requiere combinar información de diferentes partes del documento para inferir la respuesta.
    4. Unanswerable: La pregunta no puede ser respondida con el contexto proporcionado.


    REGLAS PARA ground_truth (MUY IMPORTANTE):
    - Debe ser una respuesta COMPLETA y AUTOCONTENIDA: alguien que solo lea el ground_truth debe entender la respuesta sin necesitar contexto adicional.
    - Debe contener todos los hechos clave necesarios para evaluar una respuesta de RAG (fechas, porcentajes, nombres, condiciones).
    - NUNCA copies códigos en bruto ni texto de tablas del contexto (por ejemplo, '14, 3 = ...' o 'ASI Natura 103000361').
    - NUNCA uses 'sí', 'no' o palabras sueltas como respuesta.
    - Debe ser CONCISA pero INFORMATIVA con los datos exactos (nombres, porcentajes, fechas, créditos).
    - Escribe siempre en prosa continua. NUNCA uses bullet points, listas numeradas ni markdown.
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

    - student_profile: Perfil del estudiante que haría esta pregunta 
    (ej: "estudiante de primer año", "estudiante en periodo de matrícula", 
    "estudiante preparando el TFM")

    Para preguntas Unanswerable:
    - ground_truth: No lo se
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
        "course_name": course_name,
        "distribution_str": distribution_str
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
    NUM_QUESTIONS = 20

    # ==========================================
    # 0. DIRECTORIO DEL SCRIPT (ROBUSTO)
    # ==========================================
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    print(f"📁 Guardando resultados en: {script_dir}")

    # ==========================================
    # 1. LISTA MANUAL DE PDFs
    # ==========================================
    pdf_files = [
        Path("/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Sistemas de Planificación.pdf"),
        Path("/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Probabilidades y Estadística II.pdf"),
        Path("/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Investigación Operativa.pdf"),
        Path("/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Lógica.pdf"),
        Path("/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Minería de Datos.pdf"),
        Path("/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Programación Declarativa, Lógica y Restricciones.pdf"),
        Path("/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Reconocimiento de Formas.pdf"),
    ]

    print(f"📂 Procesando {len(pdf_files)} documentos...")

    all_datasets = []

    # ==========================================
    # 2. LOOP PRINCIPAL
    # ==========================================
    for pdf_path in pdf_files:

        print(f"\n==============================")
        print(f"📄 Procesando: {pdf_path.name}")
        print(f"==============================")

        if not pdf_path.exists():
            print(f"❌ No existe: {pdf_path}")
            continue

        # ------------------------------------------
        # A. LOAD
        # ------------------------------------------
        try:
            loader = DoclingLoader(
                file_path=str(pdf_path),
                export_type=ExportType.DOC_CHUNKS,
            )
            docs = loader.load()
            full_text = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            continue

        print(f"\n--- TEXT PREVIEW (first 500 chars) ---")
        print(full_text[:500])

        # ------------------------------------------
        # B. EXTRACT SCHEMA
        # ------------------------------------------
        try:
            extracted_schema = extract_schema(full_text)
        except Exception as e:
            print(f"❌ Error extracting schema: {e}")
            continue

        print("\n--- EXTRACTED SCHEMA ---")
        print(extracted_schema.model_dump_json(indent=2))

        # ------------------------------------------
        # C. GENERATE DATASET
        # ------------------------------------------
        try:
            course_name = clean_course_name(extracted_schema.course_name)

            dataset = generate_questions(
                extracted_schema,
                course_name=course_name,
                num_questions=NUM_QUESTIONS
            )
        except Exception as e:
            print(f"❌ Error generating dataset: {e}")
            continue

        # Añadir source_document
        for pair in dataset.questions:
            pair.source_document = pdf_path.name

        # ------------------------------------------
        # D. GUARDAR DATASET INDIVIDUAL
        # ------------------------------------------
        single_output = script_dir / f"dataset_{pdf_path.stem}.json"

        single_data = {
            "source_document": pdf_path.name,
            "course_name": course_name,
            "num_questions": len(dataset.questions),
            "questions": [q.model_dump() for q in dataset.questions]
        }

        try:
            with open(single_output, "w", encoding="utf-8") as f:
                json.dump(single_data, f, ensure_ascii=False, indent=4)

            print(f"💾 Guardado individual: {single_output}")
        except Exception as e:
            print(f"❌ Error guardando individual: {e}")
            continue

        # Añadir al global
        all_datasets.append(single_data)

    # ==========================================
    # 3. GUARDAR DATASET GLOBAL
    # ==========================================
    global_output = script_dir / "dataset_global.json"

    try:
        with open(global_output, "w", encoding="utf-8") as f:
            json.dump(all_datasets, f, ensure_ascii=False, indent=4)

        print(f"\n🎉 Dataset global guardado en: {global_output}")
        print(f"📊 Total datasets generados: {len(all_datasets)}")

    except Exception as e:
        print(f"❌ Error guardando dataset global: {e}")