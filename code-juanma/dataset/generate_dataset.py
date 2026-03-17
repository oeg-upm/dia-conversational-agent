import json
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_docling import DoclingLoader

# ==========================================
# 1. DATA STRUCTURES DEFINITION (PYDANTIC)
# ==========================================

# Structure for step 1, the course guide schema
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

# Structure for step 2, the generated Q&A pair
class QAPair(BaseModel):
    question: str = Field(description="The generated question simulating a student's query (in Spanish).")
    ground_truth: str = Field(description="The ideal, factual answer based strictly on the course guide (in Spanish).")
    question_type: str = Field(description="Category: 'Factual', 'Summarization', 'Multi-hop Reasoning', or 'Unanswerable'.")
    student_profile: str = Field(description="Simulated profile: 'Freshman', 'Senior', 'Formal tone', 'Informal tone', etc.")

class QADataset(BaseModel):
    questions: List[QAPair]

# ==========================================
# 2. LOCAL LLM CONFIGURATION
# ==========================================
# Using the same backend configuration (LM Studio / Llama 3)
# Note: Extractor uses low temperature for facts, Generator uses higher temperature for creativity.

llm_extractor = ChatOpenAI(
    model="llama-3.2-3b-instruct",
    base_url="http://localhost:1234/v1",
    api_key="not_required",
    temperature=0.1 
)

llm_generator = ChatOpenAI(
    model="llama-3.2-3b-instruct",
    base_url="http://localhost:1234/v1",
    api_key="not_required",
    temperature=0.7 
)

# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================

def extract_schema(guide_text: str) -> CourseGuideSchema:
    """Step 1: Extracts pure facts from the full course guide text."""

    print("-> Extracting schema from the course guide...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic assistant. Your task is to read a university course guide and extract the key information in a structured and concise manner. 
Ignore fluff and keep the pure facts.
IMPORTANT: Extract the information and write your output in Spanish, as the original text is in Spanish."""),
        ("human", "Here is the course guide text:\n\n{text}\n\nExtract the information according to the required schema.")
    ])
    
    # Force the LLM to return a JSON matching the CourseGuideSchema
    extractor = prompt | llm_extractor.with_structured_output(CourseGuideSchema)
    schema = extractor.invoke({"text": guide_text})
    return schema

def generate_questions(schema: CourseGuideSchema, num_questions: int = 15) -> QADataset:
    """Step 2: Generates diverse questions based on the extracted schema."""

    print(f"-> Generating a diverse dataset of {num_questions} questions...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced university student simulator.
Based on the provided course data, generate {num_questions} question and answer (Q&A) pairs.

LANGUAGE REQUIREMENT:
The 'question' and 'ground_truth' fields MUST be generated in Spanish.

TAXONOMY RULES (Distribute the questions among these types):
1. Factual: Direct questions about specific facts (dates, names, office locations).
2. Summarization: Questions asking to explain a whole section (e.g., how the evaluation system works).
3. Multi-hop Reasoning: Questions that require crossing different data points (e.g., "If I work mornings, can I attend tutoring sessions?").
4. Unanswerable: Plausible questions whose answers are NOT in the data. The ground_truth must explicitly state that the course guide does not provide this information.

DIVERSITY RULES (Profiles):
Apply different styles to the questions: 'Freshman' (confused/lost), 'Senior' (direct/technical), 'Informal tone' (like a WhatsApp message), 'Formal tone' (like an email to the professor).
"""),
        ("human", "COURSE DATA:\n{schema_json}\n\nGenerate the dataset now.")
    ])
    
    schema_json = schema.model_dump_json(indent=2)
    
    generator = prompt | llm_generator.with_structured_output(QADataset)
    dataset = generator.invoke({
        "num_questions": num_questions,
        "schema_json": schema_json
    })
    
    return dataset

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Path to your course guide PDF
    pdf_path = "../../../DocumentosRAG/Guías aprendizaje/Curso 2020_2021/Grado/Grado en Ingeneiría Informática/Sistemas de Planificación.pdf"
    
    # A. Load PDF
    print(f"Loading PDF: {pdf_path}")
    try:
        loader = DoclingLoader(file_path=pdf_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error loading PDF (using fallback test text): {e}")
        full_text = ""

    print("\n--- FULL TEXT EXTRACTED FROM PDF ---")
    print(full_text)

    # B. Step 1: schema extraction
    extracted_schema = extract_schema(full_text)
    print("\n--- EXTRACTED SCHEMA ---")
    print(extracted_schema.model_dump_json(indent=2))

    # C. Step 2: dataset Generation
    # Requesting 20 questions
    final_dataset = generate_questions(extracted_schema, num_questions=20)
    
    # D. Save results
    output_file = "dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset.model_dump(), f, ensure_ascii=False, indent=4)
        
    print(f"\nProcess completed! Generated {len(final_dataset.questions)} questions and saved to '{output_file}'.")