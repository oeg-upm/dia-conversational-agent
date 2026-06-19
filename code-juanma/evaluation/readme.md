# Evaluación del sistema RAG

El sistema de evaluación calcula las siguientes métricas de rendimiento y seguridad:

* **Eje de recuperación:**
  * `Context Precision`: relevancia de los fragmentos recuperados.
  * `Context Recall`: capacidad para recuperar toda la información necesaria.
* **Eje de generación:**
  * `Faithfulness`: tasa de alucinación (adherencia estricta al contexto).
  * `Answer Relevancy`: relevancia retórica y completitud de la respuesta.
  * `Answer Similarity` & `Answer Correctness`: precisión conceptual frente a la respuesta esperada (*ground truth*).
* **Eje de seguridad:**
  * `Safe Behavior`: capacidad del modelo para rechazar consultas fuera de dominio (*Out of scope*) o ambiguas.

## Estructura de los experimentos

1. **H1a - Impacto de los embeddings:** evaluación de modelos de representación vectorial (e.g., `octen-4b`, `bge-m3`, `harrier-0.6b`).
2. **H1b - Estrategias de recuperación:** comparativa entre *Naive RAG* (recuperación lineal) frente a estrategias de expansión semántica (*Multi-Query* y *RAG-Fusion* con *RRF*).
3. **H1c - Ingeniería de prompts:** análisis de la contención generativa mediante *Zero-shot* vs. *Few-shot prompting*.
4. **H2 - Impacto de los LLMs:** comparativa entre paradigmas de inferencia (*Instruct* vs. *Reasoning models*) y escalas paramétricas (desde 8B hasta 32B), evaluando modelos como `Qwen-2.5-32B`, `Gemma-3-12B`, y `DeepSeek-R1-14B`.


## Estructura de carpetas

* `h1a/`: experimentos del impacto de los embeddings.
* `h1b/`: experimentos de las estrategias de recuperación.
* `h1c/`: experimentos de la ingeniería de prompts.
* `h2/`: experimentos del impacto de los LLMs.

En cada carpeta, se encuentran las siguientes subcarpetas: 
- `datasets/`: conjunto de datos para la evaluación.
- `results/`: resultados de la evaluación. Están tanto los archivos de cada evaluación como las medias de cada archivo para cada métrica. Además, hay dos archivos de python:
    - `mean.py`: calcula las medias de cada métrica para cada archivo.
    - `results.py`: crea un fichero csv que agrupa las medias calculadas.
    - `results_taxonomy.py`: crea un fichero csv que agrupa las medias calculadas por taxonomía (no está en el experimento h1a debido a su poca relevancia).