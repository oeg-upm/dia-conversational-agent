# RAG Evaluation Dataset

Evaluation dataset for assessing a RAG-based chatbot that allows students to ask information about the master's programme. Built to be used with [RAGAS](https://docs.ragas.io/) and [DeepEval](https://docs.confident-ai.com/).

This repository contains the evaluation dataset and schema specification for a RAG-based chatbot that allows students to ask information about the master's programme. 

The dataset aims to meet:

1. **Framework compatibility**: contains fields which map directly to the required input fields of RAGAS and DeepEval evaluation method. 
2. **Academic rigour**: the schema takes into account LLM evaluation standards.

## Research Background

### Evaluation Frameworks

**RAGAS** ([Es, J. et al., 2023](https://arxiv.org/abs/2309.15217)) decomposes RAG quality into four orthogonal metrics, each targeting a different failure mode:

| Metric | What it measures | Failure mode caught |
|---|---|---|
| `faithfulness` | Is the answer grounded in the retrieved contexts? | Hallucination |
| `answer_relevancy` | Does the answer address the question? | Verbosity / off-topic response |
| `context_precision` | Are retrieved chunks ranked correctly? | Retriever ranking quality |
| `context_recall` | Were the right chunks retrieved at all? | Retriever coverage |
| `answer_correctness` | Does the answer match the ground truth? | Factual incorrectness |

**DeepEval** provides a testing-framework-style interface and includes G-Eval, a reference-based metric powered by an LLM judge with a customisable rubric. The key metrics it uses cover the same evaluation aspects as the ones in RAGAS `AnswerRelevancy`, `Faithfulness`, `ContextualPrecision`, `ContextualRecall`, and `GEval` (correctness).

### Field-to-Metric Mapping

The table below shows which fields each metric requires.

| Metric | Framework | `question` | `answer` | `contexts` | `ground_truth` | `reference_contexts` |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `faithfulness` | RAGAS | ✅ | ✅ | ✅ | — | — |
| `answer_relevancy` | RAGAS | ✅ | ✅ | ✅ | — | — |
| `context_precision` | RAGAS | ✅ | — | ✅ | ✅ | — |
| `context_recall` | RAGAS | — | — | ✅ | — | ✅ |
| `answer_correctness` | RAGAS | — | ✅ | — | ✅ | — |
| `AnswerRelevancy` | DeepEval | ✅ | ✅ | ✅ | — | — |
| `Faithfulness` | DeepEval | — | ✅ | ✅ | — | — |
| `ContextualPrecision` | DeepEval | ✅ | — | ✅ | ✅ | — |
| `ContextualRecall` | DeepEval | — | ✅ | — | — | ✅ |
| `GEval (correctness)` | DeepEval | ✅ | ✅ | — | ✅ | — |

---

### Academic References

[To be explored...]

---

## Dataset Schema

### Field Definitions

#### 1. Framework Required Fields

Both RAGAS and DeepEval ask for these fields as the minimum unit for evaluation. You can check the specific base schemas:
  - DeepEval: [LLMTestCase](https://deepeval.com/docs/evaluation-test-cases) and [Golden Dataset](https://deepeval.com/docs/evaluation-datasets#create-an-evaluation-dataset)
  - RAGAS: [Evaluation Datasets](https://docs.ragas.io/en/stable/concepts/datasets/#dataset-structure)

| Field | Type | Description |
|---|---|---|
| `question` | `string` | The student's natural language query. Should reflect realistic, informal phrasings including ambiguous or abbreviated questions. |
| `answer` | `string` | The RAG system's generated response.|
| `contexts` | `list[string]` | The document chunks retrieved and passed to the LLM at inference time. One string per chunk. Preserves the exact context window seen by the generator. |
| `ground_truth` | `string` | Ideal answer (concise and factually complete). |
| `reference_contexts` | `list[string]` | The document chunk(s) that were used to generate the answer and should have been retrieved.|

#### 2. Source traceability

For mapping the system answers to the original documents. 

| Field | Type | Description |
|---|---|---|
| `source_document` | `string` | Filename or identifier of the source PDF/document (e.g., `plan_estudios_2026.pdf`). Required for debugging retrieval failures and for academic citation. |
| `source_document_metadata` | `dict` | Includes relevant metadata such as author, department, subject, year.|
| `chunk_id` | `string` | Unique identifier for the specific chunk within the document (e.g., `plan_estudios_2026_p12`). For further retrieval analysis. |
| `query_id` | `string` | Unique identifier for the question (for efficient lookup when analysing failure/succes cases.) |

#### 3. Question metadata

For filtered analysis of the system behaviour. These are open to experimental design (what aspects or subsets of the dataset are the target to study). 

| Field | Type | Description |
|---|---|---|
| `question_type` | `enum` | Query category. Posible allowed values (could be any other decided when designing experiments): `factual`, `procedural`, `comparative`, `out_of_scope`, `ambiguous`. Allows metric breakdown by difficulty class. |
| `topic` | `string` | Thematic area within the programme (e.g., `plan_de_estudios`, `matricula`, `tfm`, `profesorado`). Enables per-topic performance analysis. |
| `difficulty` | `enum` | Subjective difficulty: `easy`, `medium`, `hard`. Based on the number of sources that must be consulted/synthesised and how implicit or open the answer is. |
| `generation_method` | `enum` | How the question was created: `human_authored`, `llm_generated`, `llm_then_human_verified`.|

> The field `question_type` is a very wide/open category at the moment, it could be divided into more specific category fields instead of having that many posible values. 

#### 4. Optional/Additional fields

Ideas that could be added for further system behaviour study.

| Field | Type | Description |
|---|---|---|
| `language` | `enum` | Language of the query: `es`, `en`.|
| `conversation` | `dict` | History of previous messages user/system (for multiturn)|

---

## Example Record

```json
{
  "sample_id":           "q_042",
  "generation_method":   "llm_then_human_verified",
  "language":            "es",

  "question":            "¿Cuántos créditos tiene la asignatura de Aprendizaje Automático?",
  "answer":              "La asignatura de Aprendizaje Automático tiene 6 créditos ECTS y es de carácter obligatorio en el primer semestre.",
  "contexts": [
    "Aprendizaje Automático — 6 ECTS — Obligatoria — Semestre 1. La asignatura introduce los fundamentos del aprendizaje supervisado y no supervisado...",
    "Las asignaturas obligatorias del máster suman un total de 48 créditos ECTS distribuidos en dos semestres."
  ],

  "ground_truth":        "La asignatura tiene 6 créditos ECTS y es de carácter obligatorio.",
  "reference_contexts": [
    "Aprendizaje Automático — 6 ECTS — Obligatoria — Semestre 1. La asignatura introduce los fundamentos del aprendizaje supervisado y no supervisado..."
  ],

  "source_document":     "plan_estudios_master_2026.pdf",
  "chunk_id":            "plan_estudios_master_2026_p12",

  "question_type":       "factual",
  "topic":               "plan_de_estudios",
  "difficulty":          "easy"
}
```

**Out-of-scope example** - testing behaviour when no relevant context is found:

```json
{
  "sample_id":           "q_087",
  "generation_method":   "human_authored",
  "language":            "es",

  "question":            "¿Puedo convalidar asignaturas de otro máster europeo?",
  "answer":              "Lo siento, no tengo información sobre el procedimiento de convalidación de asignaturas de programas externos. Te recomiendo contactar directamente con la secretaría académica.",
  "contexts": [
    "El plan de estudios del máster contempla 60 créditos ECTS distribuidos entre asignaturas obligatorias, optativas y el TFM..."
  ],

  "ground_truth":        "El sistema no dispone de información sobre convalidaciones con programas externos. El estudiante debe contactar con secretaría.",
  "reference_contexts":  [],

  "source_document":     "N/A",
  "chunk_id":            "N/A",

  "question_type":       "out_of_scope",
  "topic":               "tramites_administrativos",
  "difficulty":          "hard"
}
```

---

## Question Taxonomy

The dataset targets the following distribution across question types. This mirrors the stratified approach used in TyDi QA and TREC evaluation tracks, adapted to the academic chatbot domain.

| Type | Description | Example |
|---|---|---|
| `factual` | Single-hop fact retrieval from one chunk | *"¿Cuántos ECTS tiene el TFM?"* |
| `procedural` |  Step-by-step process questions | *"¿Cómo me matriculo en el segundo semestre?"* |
| `comparative` | Requires synthesising multiple chunks | *"¿Qué asignaturas tienen más créditos, las obligatorias o las optativas?"* |
| `out_of_scope` | Answer not present in indexed documents | *"¿Puedo convalidar con un máster europeo?"* |
| `ambiguous` | Underspecified queries with multiple valid interpretations | *"¿Cuándo son los exámenes?"* |

> The `out_of_scope` and `ambiguous` categories are deliberately included to stress-test system robustness. A hallucinating system will score poorly on `faithfulness` for `out_of_scope` items; an overconfident system will fail `answer_relevancy` on `ambiguous` ones.

---

## Dataset Building Methodology

Proposed: **semi-automatic approach** combining LLM-assisted generation with human verification, following the methodology proposed in ARES (Saad-Falcon et al., 2023).

### Step 1 - Document ingestion

Source documents (programme syllabi, enrollment guides, faculty pages) were chunked using a fixed-size strategy with overlap:
- Chunk size: 512 tokens
- Overlap: 64 tokens
- Metadata attached per chunk: `source_document`, `page_number`, `chunk_index`

### Step 2 - Synthetic question generation

For each source chunk, an LLM was prompted to generate candidate questions of each `question_type`. The prompt template used:

```
Given the following document excerpt, generate one {question_type} question 
that a master's student might realistically ask. The question should be 
answerable from the excerpt alone. Return only the question, no preamble.

Excerpt: {chunk_text}
```

`out_of_scope` questions were generated separately by prompting the model to produce plausible but unanswerable questions about topics adjacent to the programme.

### Step 3 - Human verification

All LLM-generated questions were reviewed by a human annotator who:
- Confirmed the question is naturally phrased and realistic
- Verified the `ground_truth` answer is factually correct
- Confirmed the `reference_contexts` field points to the correct chunk(s)
- Assigned `difficulty` rating

Questions failing any of these checks were discarded or rewritten. Final `generation_method` values:

| Value | Meaning |
|---|---|
| `human_authored` | Written directly by a human annotator |
| `llm_generated` | LLM output used as-is (rare, only for clearly unambiguous factual items) |
| `llm_then_human_verified` | LLM-generated question reviewed and approved/edited by a human |
---