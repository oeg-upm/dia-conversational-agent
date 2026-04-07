import json
import os
import warnings
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_docling import DoclingLoader

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ==========================================
# 1. DATA STRUCTURES DEFINITION (PYDANTIC)
# ==========================================

class CourseGuideSchema(BaseModel):
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
    """A single evaluation case compatible with RAGAS/DeepEval."""
    model_config = ConfigDict(extra='ignore')
    
    question: str = Field(description="The student's query (in Spanish).")
    ground_truth: str = Field(description="The ideal factual answer (in Spanish).")
    reference_contexts: List[str] = Field(description="Exact text snippets from the source.")
    question_type: str = Field(description="Taxonomy: factual, procedural, comparative, out_of_scope, or ambiguous.")
    topic: str = Field(description="Thematic area (e.g., evaluation, syllabus).")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard.")
    source_document: str = Field(description="Filename of the source PDF.")

class QADataset(BaseModel):
    """Collection of QA pairs used for batching."""
    questions: List[QAPair]

# ==========================================
# 2. LLM CONFIGURATION
# ==========================================

LLM_SETTINGS = {
    "model": "qwen2.5:32b", 
    "base_url": "http://100.83.249.109:5000/v1",
    "api_key": "not_required",
    "timeout": 600,
}

llm_extractor = ChatOpenAI(**LLM_SETTINGS, temperature=0.1)
llm_generator = ChatOpenAI(**LLM_SETTINGS, temperature=0.7)

# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================

def extract_course_schema(guide_text: str) -> CourseGuideSchema:
    """Step 1: Clean raw PDF text into a structured schema."""
    print("-> Extracting structured schema from course guide...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic assistant. Extract key facts in Spanish."),
        ("human", "Course Guide Text:\n\n{text}\n\nExtract information according to the schema.")
    ])
    
    extractor = prompt | llm_extractor.with_structured_output(CourseGuideSchema)
    return extractor.invoke({"text": guide_text})

def generate_evaluation_dataset(schema: CourseGuideSchema, filename: str, total_count: int = 20) -> List[dict]:
    """Step 2: Generate QA pairs in batches."""
    print(f"-> Starting batch generation for {total_count} samples...")
    
    final_questions = []
    batch_size = 5 # Small batches are more stable
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert RAG evaluation annotator. 
Generate a 'Golden Dataset' in Spanish following this taxonomy:
1. factual: Single-hop facts.
2. procedural: Step-by-step processes.
3. comparative: Synthesizing multiple sections.
4. out_of_scope: Plausible questions NOT in text (empty reference_contexts).
5. ambiguous: Vague queries.

IMPORTANT: 'reference_contexts' must contain exact Spanish snippets from the data."""),
        ("human", "SOURCE FILE: {filename}\nDATA:\n{schema_json}\nGenerate {batch_count} samples.")
    ])
    
    generator = prompt | llm_generator.with_structured_output(QADataset)
    schema_json = schema.model_dump_json(indent=2)

    for i in range(0, total_count, batch_size):
        current_batch_size = min(batch_size, total_count - i)
        print(f"   - Processing batch {i//batch_size + 1} ({current_batch_size} questions)...")
        
        try:
            batch_result = generator.invoke({
                "schema_json": schema_json,
                "batch_count": current_batch_size,
                "filename": filename
            })
            for q in batch_result.questions:
                q.source_document = filename # Ensure traceability metadata
                final_questions.append(q.model_dump())
            print(f"   [OK] Batch completed. Total count: {len(final_questions)}")
        except Exception as e:
            print(f"   [ERROR] Batch failed: {e}")
            
    return final_questions

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    pdf_filename = "Sistemas de Planificación.pdf"
    pdf_path = f"../../../DocumentosRAG/Guías aprendizaje/Curso 2020_2021/Grado/Grado en Ingeneiría Informática/{pdf_filename}"
    
    print(f"Loading PDF: {pdf_path}")
    try:
        loader = DoclingLoader(file_path=pdf_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Loading error: {e}")
        full_text = "Error loading text..."

    # A. Information Extraction
    course_data = extract_course_schema(full_text)
    print("\n--- SCHEMA EXTRACTED SUCCESSFULLY ---")

    # B. Dataset generation (batching mode)
    # This will run 4 iterations of 5 questions each
    qa_list = generate_evaluation_dataset(course_data, pdf_filename, total_count=20)
    
    # C. Save results
    output_file = "rag_dataset_v2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=4)
        
    print(f"\nProcess completed! {len(qa_list)} cases saved to '{output_file}'.")