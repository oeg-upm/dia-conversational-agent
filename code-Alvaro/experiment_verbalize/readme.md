# Experimento 1: Verbalización de tablas en un sistema RAG

Este experimento analiza el impacto de la verbalización de documentos en el rendimiento de un sistema RAG aplicado a guías docentes universitarias.

El objetivo principal es comparar dos configuraciones equivalentes del sistema RAG:

- `rag_code_base`: sistema RAG construido a partir de los documentos originales.
- `rag_code_verb`: sistema RAG construido a partir de los documentos verbalizados.

La comparación se realiza utilizando los mismos datasets de evaluación, los mismos modelos y la misma configuración general del sistema, modificando únicamente el tipo de documentos indexados en la base vectorial.


---

## 1. Configuraciones del RAG

Para reproducir el experimento se utilizan dos instancias del sistema RAG.

### 1.1. RAG base

El RAG base utiliza los documentos originales de las guías docentes.

```text
Carpeta: rag_code_base
Puerto de ChromaDB: 8002
Collection name: basic_rag_2
Documentos utilizados: PDFs originales
```

### 1.2. RAG verbalizado

El RAG verbalizado utiliza los documentos generados por el script `verbalize_documents.py`.

```text
Carpeta: rag_code_verb
Puerto de ChromaDB: 8001
Collection name: basic_rag
Documentos utilizados: PDFs verbalizados
```

Es importante que ambos RAGs mantengan la misma configuración general para que la comparación sea justa. La diferencia principal entre ambos debe ser únicamente el conjunto de documentos indexados.

---

## 2. Ejecución inicial del RAG

Antes de realizar la ingesta de documentos, se debe levantar el sistema RAG correspondiente siguiendo las instrucciones del README propio del RAG.

Por tanto, el primer paso consiste en ejecutar el RAG con las especificaciones indicadas en su README.

Para el experimento con documentos originales se debe levantar:

```text
rag_code_base
```

Para el experimento con documentos verbalizados se debe levantar:

```text
rag_code_verb
```

En ambos casos, el servicio de la API del RAG debe estar disponible en:

```text
http://localhost:9000
```

El endpoint utilizado para la ingesta de documentos es:

```text
POST /upload
```

---

## 3. Verbalización de documentos

El script encargado de generar los documentos verbalizados es:

```text
verbalize_documents.py
```

Este script procesa los documentos completos de las guías docentes. Su objetivo es transformar elementos semiestructurados, especialmente tablas, en texto en lenguaje natural.

El proceso general es el siguiente:

1. Leer los PDFs originales de las guías docentes.
2. Convertir cada documento a Markdown.
3. Detectar tablas dentro del contenido.
4. Enviar cada tabla al modelo de lenguaje.
5. Sustituir la tabla original por una verbalización textual.
6. Exportar el documento verbalizado de nuevo a PDF.

Los documentos resultantes se guardan en:

```text
experiment_verbalize/docs_verb/
```

Estos PDFs serán posteriormente utilizados para construir el RAG verbalizado.

---

## 4. Configuración del modelo de lenguaje

El experimento utiliza un modelo de lenguaje desplegado en un servidor externo de la universidad.

Por este motivo, antes de ejecutar los scripts es necesario revisar la URL del modelo dentro del código.

Ejemplo de configuración:

```python
llm_verbalizer = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://URL_DEL_SERVIDOR/v1",
    api_key="not_required",
    temperature=0.2,
    timeout=300
)
```

En caso de ejecutar el experimento en otro entorno, se debe modificar `base_url` para que apunte al servidor donde esté desplegado el modelo.

Por ejemplo:

```python
base_url="http://localhost:11434/v1"
```

o bien:

```python
base_url="http://IP_DEL_SERVIDOR:PUERTO/v1"
```

Esta modificación es necesaria tanto en los scripts de verbalización como en aquellos scripts de generación o evaluación que hagan llamadas al modelo de lenguaje.

---

## 5. Ingesta de documentos

Una vez levantado el RAG correspondiente, se deben subir los documentos mediante el script:

```text
upload_docs.py
```

Este script envía los PDFs al endpoint `/upload` del sistema RAG:

```python
API_URL = "http://localhost:9000/upload"
```

---

### 5.1. Ingesta de documentos originales

Para construir el RAG base, se deben utilizar los PDFs originales:

```python
files = [
    "Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Sistemas de Planificación.pdf",
    "Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Probabilidades y Estadística II.pdf",
    "Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Investigación Operativa.pdf",
    "Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Lógica.pdf",
    "Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Minería de Datos.pdf",
    "Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Programación Declarativa, Lógica y Restricciones.pdf",
    "Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Reconocimiento de Formas.pdf",
]
```
Esta ingesta debe hacerse con el RAG base levantado:

```text
ChromaDB port: 8002
Collection name: basic_rag_2
```

Después, se ejecuta:

```bash
python upload_docs.py
```

---

### 5.2. Ingesta de documentos verbalizados

Para construir el RAG verbalizado, se debe comentar la lista anterior y descomentar la lista de documentos verbalizados:

```python
files = [
    "docs_verb/Sistemas de Planificación.pdf",
    "docs_verb/Probabilidades y Estadística II.pdf",
    "docs_verb/Investigación Operativa.pdf",
    "docs_verb/Lógica.pdf",
    "docs_verb/Minería de Datos.pdf",
    "docs_verb/Programación Declarativa, Lógica y Restricciones.pdf",
    "docs_verb/Reconocimiento de Formas.pdf",
]
```

Esta ingesta debe hacerse con el RAG verbalizado levantado:

```text
ChromaDB port: 8001
Collection name: basic_rag
```

Después, se ejecuta:

```bash
python upload_docs.py
```

---

---

## 6. Datasets de evaluación

Una vez realizada la ingesta de documentos en el RAG correspondiente, se pueden ejecutar los experimentos con los datasets de evaluación.

Se utilizan principalmente dos datasets:

```text
dataset_general.json
dataset_tables.json
```

### 6.1. Dataset general

El dataset general contiene preguntas variadas sobre las guías docentes. Incluye preguntas factuales, preguntas de resumen, preguntas multi-hop y preguntas no respondibles.

Ruta recomendada:

```text
experiment_verbalize/datasets/dataset_general.json
```

### 6.2. Dataset de tablas

El dataset de tablas contiene preguntas diseñadas específicamente para evaluar información procedente de tablas, como porcentajes de evaluación, actividades, cronogramas, prácticas, semanas o condiciones de evaluación.

Ruta recomendada:

```text
experiment_verbalize/datasets/dataset_tables.json
```

---

## 7. Ejecución del experimento

Para ejecutar la evaluación se utiliza el script:

```text
run_evaluation.py
```

Antes de lanzarlo, se debe comprobar que apunta al RAG correcto.

Además, si se desea evaluar únicamente un tipo concreto de pregunta, se debe modificar la variable `QUESTION_TYPE` dentro de `run_evaluation.py`.

Por ejemplo:

```python
QUESTION_TYPE = ""
```

permite evaluar todas las preguntas del dataset.

En cambio, para evaluar solo un tipo específico de pregunta, se puede indicar el tipo correspondiente:

```python
QUESTION_TYPE = "Factual"
```

```python
QUESTION_TYPE = "Summarization"
```

```python
QUESTION_TYPE = "Multi-hop Reasoning"
```

```python
QUESTION_TYPE = "Unanswerable"
```

El valor de `QUESTION_TYPE` debe coincidir con el campo de tipo de pregunta utilizado en el dataset de evaluación. Si se deja vacío, se evalúa el dataset completo.

---

### 7.1. Evaluación del RAG base

Para evaluar el RAG construido con documentos originales, la configuración debe apuntar a:

```text
ChromaDB port: 8002
Collection name: basic_rag_2
```

Además, debe estar levantado el RAG de la carpeta:

```text
rag_code_base
```

La ejecución sería:

```bash
python run_evaluation.py
```

---

### 7.2. Evaluación del RAG verbalizado

Para evaluar el RAG construido con documentos verbalizados, la configuración debe apuntar a:

```text
ChromaDB port: 8001
Collection name: basic_rag
```

Además, debe estar levantado el RAG de la carpeta:

```text
rag_code_verb
```

La ejecución sería:

```bash
python run_evaluation.py
```

---

## 8. Métricas utilizadas

La evaluación se realiza mediante RAGAS, utilizando las siguientes métricas:

```text
faithfulness
answer_relevancy
context_recall
context_precision
```

Estas métricas permiten analizar tanto la calidad de la respuesta generada como la calidad de los contextos recuperados por el sistema RAG.

---

## 9. Resultados esperados

Los resultados de cada ejecución se almacenan en archivos CSV dentro de la carpeta de resultados.

Ejemplo:

```text
results/
├── evaluation_base_general.csv
├── evaluation_base_table.csv
├── evaluation_verbalizado_general.csv
├── evaluation_verbalizado_table.csv
├── evaluation_summary_general.csv
└── evaluation_summary_table.csv
```

La comparación principal del experimento se realiza entre:

```text
RAG base + dataset_general
RAG verbalizado + dataset_general
RAG base + dataset_tables
RAG verbalizado + dataset_tables
```

Esta comparación permite observar si la verbalización mejora especialmente el rendimiento en preguntas que dependen de información tabular.

---

## 10. Resumen del flujo completo

El flujo completo para reproducir el experimento es:

```text
1. Levantar rag_code_base.
2. Configurar ChromaDB en el puerto 8002 con la colección basic_rag_2.
3. Ejecutar upload_docs.py con los documentos originales.
4. Ejecutar run_evaluation.py con dataset_general.json y dataset_tables.json.
5. Guardar los resultados del RAG base.

6. Levantar rag_code_verb.
7. Configurar ChromaDB en el puerto 8001 con la colección basic_rag.
8. Ejecutar upload_docs.py con los documentos verbalizados.
9. Ejecutar run_evaluation.py con dataset_general.json y dataset_tables.json.
10. Guardar los resultados del RAG verbalizado.

11. Comparar los resultados obtenidos en ambos sistemas.
```

---

## 11. Advertencias importantes

### 11.1. Rutas locales

El código original utiliza rutas absolutas del entorno local de desarrollo. Por tanto, antes de ejecutar el experimento en otro equipo, se deben modificar las rutas de los documentos en:

```text
upload_docs.py
verbalize_documents.py
```

### 11.2. URL del modelo de lenguaje

El modelo de lenguaje se encuentra desplegado en un servidor externo de la universidad. Si se ejecuta el experimento en otro entorno, es obligatorio modificar la URL del modelo.

Se debe revisar especialmente:

```text
base_url
ollama_url
```

### 11.3. Puerto y colección de ChromaDB

Es importante no mezclar las colecciones del RAG base y del RAG verbalizado.

La configuración correcta es:

```text
RAG base:
- ChromaDB port: 8002
- Collection name: basic_rag_2

RAG verbalizado:
- ChromaDB port: 8001
- Collection name: basic_rag
```

### 11.4. Limpieza de colecciones

Antes de repetir una ingesta, se recomienda comprobar que la colección de ChromaDB está vacía o eliminar la colección anterior. De lo contrario, podrían mezclarse documentos de ejecuciones anteriores y alterar los resultados.

### 11.5. Comparación justa

Para que la comparación sea válida, deben mantenerse constantes:

```text
Modelo generativo
Modelo de embeddings
Número de documentos
Datasets de evaluación
Número de ejecuciones
Configuración del recuperador
Métricas de evaluación
```

La única diferencia entre ambos experimentos debe ser el uso de documentos originales o documentos verbalizados.

---

## 12. Finalidad del experimento

Este experimento permite estudiar si transformar tablas en lenguaje natural mejora la capacidad del sistema RAG para recuperar información relevante y generar respuestas correctas.

La hipótesis principal es que la verbalización puede mejorar el rendimiento en preguntas tabulares, ya que convierte información estructurada en texto más fácilmente recuperable por el sistema. Sin embargo, también puede introducir más contenido textual, redundancia o ruido, por lo que resulta necesario comparar su efecto tanto en preguntas generales como en preguntas específicamente diseñadas sobre tablas.