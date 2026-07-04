# Documentación Técnica del Generador de Dataset de Evaluación
## Proyecto DIA — Agente Conversacional para la FI-UPM

**Guías de referencia:** AESIA #15 (§5.2.d, §5.2.g) · AESIA #07 (Datos y gobernanza)  
**Implementación de referencia:** `code-juanma/dataset/generate_dataset_V3.py`  
**Fecha de redacción:** Julio 2026

---

## 1. Propósito del sistema (Guía 15, §5.1)

El generador de dataset es un sistema auxiliar de evaluación que produce automáticamente pares pregunta-respuesta de referencia (*QA pairs*) para ser utilizados en la evaluación del sistema RAG principal. Su función es doble:

1. **Generar preguntas realistas** que un estudiante podría formular, a partir del corpus de guías docentes
2. **Obtener y registrar respuestas reales** del sistema RAG para cada pregunta generada, junto con los contextos recuperados y una respuesta de referencia (*ground truth*) generada por el LLM

El output del sistema (archivos JSON) sirve como input para los evaluadores RAGAS descritos en [04_evaluadores_documentacion_tecnica.md](04_evaluadores_documentacion_tecnica.md).

---

## 2. Arquitectura del generador (Guía 15, §5.2)

### Diagrama de flujo

```
[ChromaDB Vector Store]
        │
        ▼ get_db_chunks() — extrae N chunks aleatorios
[Chunks seleccionados]
        │
        ▼ Para cada chunk:
[LLM: llama3.1:8b o qwen2.5:32b]
  Genera: question, ground_truth, question_type,
          topic, difficulty, language
        │
        ▼
[RAG Backend: POST /chat?context=True]
  Obtiene: answer (real RAG response)
           contexts (chunks realmente recuperados)
        │
        ▼
[Objeto QAPair (Pydantic)] → JSON dataset
```

### Componentes del sistema

| Componente | Tecnología | Función |
|------------|-----------|---------|
| Fuente de chunks | ChromaDB HTTP Client | Extrae chunks reales del corpus vectorizado |
| Generador de QA | LangChain + LLM (structured output) | Genera preguntas, ground truth y metadatos |
| Cliente RAG | `requests.post` al backend | Obtiene respuesta real del sistema RAG |
| Esquema de datos | Pydantic (`QAPair`) | Validación estructural de cada registro generado |
| Serialización | JSON (UTF-8) | Persistencia del dataset generado |

---

## 3. Datos de entrada y gobernanza (Guía 07)

### 3.1 Fuente de datos primaria (corpus)

| Atributo | Valor |
|----------|-------|
| Tipo | Documentos PDF públicos |
| Origen | Guías de aprendizaje (*guías docentes*) de la Facultad de Informática, UPM |
| Acceso | Público — disponibles en la web de la facultad |
| Idioma | Español |
| Contenido | Descripciones de asignaturas, profesorado, requisitos, programas, criterios de evaluación, calendarios, bibliografías |
| Metadatos | `course` (año académico), `degree` (titulación), `category`, `source` (nombre de archivo) |

### 3.2 Análisis de sesgos del corpus (Guía 07, §4.3.8)

**Sesgos identificados:**

| Tipo de sesgo | Descripción | Impacto |
|---------------|-------------|---------|
| Sesgo de cobertura | El corpus incluye únicamente guías de la FI-UPM; preguntas sobre otras facultades o instituciones son `out_of_scope` por diseño | Intencional — delimita el alcance del sistema |
| Sesgo temporal | Las guías corresponden a cursos académicos específicos; información desactualizada puede persistir si no se re-ingestan nuevas versiones | Medio — gestionado por re-ingestión manual |
| Sesgo de idioma | Corpus 100% en español; preguntas en otros idiomas no tienen contexto de referencia | Bajo — el sistema está diseñado para uso en español |
| Sesgo de representación | Las guías de programas más grandes tienen más chunks; el muestreo aleatorio del generador puede sobrerrepresentarlos | Bajo — el muestreo es aleatorio sobre el total de chunks |

### 3.3 Procedencia de los datos de entrenamiento del generador

El generador no utiliza datos de entrenamiento propios — utiliza un LLM preentrenado (`llama3.1:8b` o `qwen2.5:32b`) que no fue fine-tuneado con datos del proyecto. Los únicos datos que procesa son los chunks del corpus de guías docentes de la UPM.

---

## 4. Esquema del dataset generado (Guía 15, §5.2.d)

### 4.1 Estructura de un registro QAPair

```json
{
  "sample_id": "42",
  "generation_method": "llm_generated",
  "language": "es",
  "question": "¿Cuáles son los criterios de evaluación para la asignatura X?",
  "answer": "[respuesta real del sistema RAG]",
  "ground_truth": "[respuesta ideal generada por el LLM]",
  "contexts": ["chunk1 recuperado por RAG", "chunk2 recuperado por RAG"],
  "reference_contexts": ["chunk original del que se generó la pregunta"],
  "source_document": "guia_asignatura_X_2024-25.pdf",
  "chunk_id": "2024-25_GII_guia_asignatura_X_2024-25.pdf_ch_12",
  "question_type": "factual",
  "topic": "plan_de_estudios",
  "difficulty": "easy"
}
```

### 4.2 Taxonomía de tipos de preguntas

La distribución de tipos de preguntas está controlada mediante pesos probabilísticos:

| Tipo | Descripción | Probabilidad |
|------|-------------|-------------|
| `factual` | Hecho explícito recuperable en una sola frase | 40% |
| `procedural` | Proceso paso a paso (e.g., procedimiento de matrícula) | 20% |
| `comparative` | Síntesis de información de múltiples fuentes | 20% |
| `out_of_scope` | Plausible pero no respondible con el corpus | 10% |
| `ambiguous` | Pregunta vaga que requiere aclaración | 10% |

*Nota: Para preguntas `out_of_scope`, el campo `source_document` se registra como `"N/A"` y `reference_contexts` como lista vacía, ya que no existe contexto de referencia aplicable.*

### 4.3 Taxonomía de temas

| Tema | Descripción |
|------|-------------|
| `plan_de_estudios` | Asignaturas, créditos, estructura del plan |
| `matricula` | Procedimientos de matrícula y admisión |
| `tfm` | Trabajo Fin de Máster |
| `profesorado` | Información sobre docentes |
| `otros` | Cualquier otra categoría |

### 4.4 Niveles de dificultad

| Dificultad | Criterio |
|-----------|---------|
| `easy` | La respuesta está explícita en una sola frase del chunk |
| `medium` | Requiere consultar múltiples partes del texto o parafrasear |
| `hard` | La respuesta es implícita, requiere síntesis o la pregunta es abierta |

---

## 5. Pipeline de generación (Guía 07, §4.2 — Preparación de datos)

### 5.1 Proceso detallado

```python
# Paso 1: Conexión a ChromaDB y extracción de chunks
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("rag_collection")
chunks = collection.get()  # todos los chunks indexados

# Paso 2: Muestreo aleatorio
random.shuffle(chunks)
selected_chunks = chunks[:n]  # n=100 por defecto

# Paso 3: Para cada chunk — generación con LLM (structured output)
record = generator.invoke({
    "chunk_text": chunk['text'],
    "metadata": json.dumps(chunk['metadata']),
    "q_type": q_type  # seleccionado por distribución de probabilidades
})

# Paso 4: Obtención de respuesta real del RAG
res = get_rag_response(record.question, chunk['metadata'])
record.answer = res['response']
record.contexts = res['context']
```

### 5.2 Prompt de generación (extracto)

El LLM recibe un system prompt detallado con las reglas de taxonomía y el siguiente prompt de usuario:

```
Context: {chunk_text}
Metadata: {metadata}
Type requested: '{q_type}'
```

La respuesta se valida mediante structured output (Pydantic `QAPair.model_validate()`), lo que garantiza que todos los campos requeridos están presentes y tienen el formato correcto.

### 5.3 Control de calidad de los datos generados

| Mecanismo | Descripción |
|-----------|-------------|
| Validación de esquema | Pydantic valida tipo y formato de cada campo antes de guardarlo |
| IDs consecutivos | `sample_id` se asigna secuencialmente tras validación exitosa |
| Metadatos de trazabilidad | Cada registro incluye `chunk_id` y `source_document` para auditoría |
| Respuesta real del RAG | El campo `answer` siempre proviene del backend real, no de una simulación |
| Manejo de errores | Registros con error de generación se omiten (try/except por chunk) |

---

## 6. Versiones del dataset producido (Guía 15, §5.8)

| Archivo | LLM generador | Descripción |
|---------|--------------|-------------|
| `datasets/rag_dataset_v1.json` | — | Primera versión básica |
| `datasets/rag_dataset_v2.json` | — | Segunda versión con mejoras de esquema |
| `datasets/rag_dataset_v3_qwen2.5_32b.json` | qwen2.5:32b | V3 con qwen2.5 32B como generador |
| `datasets/rag_dataset_v3_gemma3_27b.json` | gemma3:27b | V3 con gemma3 27B como generador |
| `datasets/rag_dataset_v3_deepseek_r1_32b.json` | deepseek-r1:32b | V3 con DeepSeek R1 32B |
| `datasets/rag_dataset_v3_llama_3-1_8b.json` | llama3.1:8b | V3 con LLaMA 3.1 8B |
| `datasets/rag_dataset_v3_ministral_3_14b.json` | ministral-3:14b | V3 con Ministral 3 14B |
| `datasets/rag_dataset_v3_gemma4_26b.json` (principal) | gemma4:26b | Dataset final seleccionado para evaluación |
| `code-andre/dataset/rag_dataset_humanized_v1.json` | Manual | Dataset humanizado manualmente (ver §7) |

### 6.1 Dataset seleccionado para el experimento principal

El dataset `rag_dataset_v3_octen_qwen2.5_V2.json` (generado con qwen2.5:32b) fue seleccionado como dataset base *original* para el experimento de humanización, por ofrecer la mejor calidad de preguntas según evaluación cualitativa del equipo.

---

## 7. Dataset humanizado (proceso manual — contexto)

El dataset humanizado (`rag_dataset_humanized_v1.json`) se generó manualmente a partir del dataset original. Este proceso no está documentado como sistema automatizado porque fue realizado por el investigador de forma manual, aplicando transformaciones de estilo:

- Conversión de preguntas formales a tono conversacional
- Introducción de ortografía informal y coloquialismos
- Adición de contexto personal ("para mi TFM", "me confundo con...")
- Conservación del contenido factual de cada pregunta

**Los 100 pares pregunta-respuesta son idénticos en contenido**, solo varía el estilo de formulación. Esta propiedad es fundamental para la validez del bootstrap pareado utilizado en la evaluación (ver [04_evaluadores_documentacion_tecnica.md](04_evaluadores_documentacion_tecnica.md)).

---

## 8. Limitaciones del generador (Guía 15, §5.7)

1. **LLM como generador:** La calidad de las preguntas generadas depende del LLM utilizado. Preguntas generadas por modelos más pequeños pueden ser menos naturales o tener errores factuales.
2. **Sesgo de chunk:** El sistema genera una pregunta por chunk seleccionado; chunks más largos o más informativos generan preguntas más ricas.
3. **Ground truth generado:** El campo `ground_truth` es una respuesta ideal generada por el mismo LLM, no una respuesta validada por expertos humanos. Esto puede introducir sesgo en métricas como `answer_correctness` que comparan contra el ground truth.
4. **Ausencia de deduplicación temática:** No existe control explícito para evitar que múltiples chunks de la misma asignatura generen preguntas similares.
5. **Sin revisión humana sistemática:** No se realizó una revisión humana exhaustiva del dataset generado. Se realizó una revisión cualitativa selectiva.

---

*Para la documentación del sistema de evaluación que consume estos datasets, ver [04_evaluadores_documentacion_tecnica.md](04_evaluadores_documentacion_tecnica.md).*
