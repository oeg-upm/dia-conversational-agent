# A RAG-Based Institutional Assistant  
https://arxiv.org/abs/2501.13880

En este paper se construye un **agente conversacional institucional** para la Universidad de São Paulo utilizando una arquitectura **RAG modular**.  

- Para la etapa de **recuperación** se emplean embeddings de **Sentence-BERT** indexados en una base de datos vectorial **FAISS**.  
- Para la **generación de respuestas** se comparan varios **LLMs**, incluyendo **GPT-3.5**, **LLaMA-3** y **Mixtral**.  
- La **orquestación del pipeline** se implementa mediante **LangChain**.

---

## Resultados principales

Se demuestra que el factor más crítico en el rendimiento de un sistema **RAG + LLM** es la **calidad del retrieval**:

> Si el retriever no recupera el chunk correcto, el LLM no puede producir una buena respuesta.

Por tanto, la **limitación principal** de la calidad de las respuestas (cuello de botella) del sistema no está en el modelo generativo, sino en la **etapa de recuperación**.

Experimentalmente, el paper muestra que cuando el chunk correcto está disponible en el contexto, la **precisión del LLM aumenta en más de 30 puntos porcentuales**.

Además, los resultados sugieren que para mejorar el retrieval es necesario **combinar**:

- **Búsqueda léxica (BM25)**
- **Búsqueda semántica por embeddings**

ya que cada una funciona mejor para distintos tipos de consultas.

---

## Tamaño de los chunks

También se discute la importancia del tamaño de los chunks:

- **Chunks pequeños (~2k tokens)** → mejores recuperaciones semánticas  
- **Chunks grandes (~8k tokens)** → introducen más ruido y degradan el retrieval

---

# RAGAS: Automated Evaluation of Retrieval Augmented Generation  
https://arxiv.org/pdf/2309.15217

Este paper es **muy importante para el proyecto**, ya que introduce un **framework para evaluar sistemas RAG**.  

Define tres dimensiones clave:

- **Faithfulness** → La respuesta está respaldada por el contexto  
- **Context relevance** → El contexto recuperado es relevante y no ruido  
- **Answer relevance** → La respuesta realmente responde a la pregunta
