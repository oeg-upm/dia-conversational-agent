## Storages open source

**MinIO**\
About: MinIO is a high-performance, S3 compatible object store, open sourced under GNU AGPLv3 license.\
[GitHub link](https://github.com/minio/minio) ⭐ 60.3k

**SeaweedFS**\
About: SeaweedFS is a fast distributed storage system for blobs, objects, files, and data lake, for billions of files! Blob store has O(1) disk seek, cloud tiering. Filer supports Cloud Drive, xDC replication, Kubernetes, POSIX FUSE mount, S3 API, S3 Gateway, Hadoop, WebDAV, encryption, Erasure Coding. Enterprise version is at seaweedfs.com.\
[GitHub link](https://github.com/seaweedfs/seaweedfs) ⭐ 30.4k

**RustFS**\
About: 🚀2.3x faster than MinIO for 4KB object payloads. RustFS is an open-source, S3-compatible high-performance object storage system supporting migration and coexistence with other S3-compatible platforms such as MinIO and Ceph.\
[GitHub link](https://github.com/rustfs/rustfs) ⭐ 22.1k

**CloudServer**\
About: Zenko CloudServer, an open-source Node.js implementation of the Amazon S3 protocol on the front-end and backend storage capabilities to multiple clouds, including Azure and Google.\
[GitHub link](https://github.com/scality/cloudserver) ⭐ 1.9k

---

## Bases de datos vectoriales

**Milvus**\
About: Milvus is a high-performance, cloud-native vector database built for scalable vector ANN search.\
[GitHub link](https://github.com/milvus-io/milvus) ⭐ 42.9k

**Faiss**\
About: A library for efficient similarity search and clustering of dense vectors.\
[GitHub link](https://github.com/facebookresearch/faiss) ⭐ 39.1k

**Qdrant**\
About: Qdrant - High-performance, massive-scale Vector Database and Vector Search Engine for the next generation of AI. Also available in the cloud.\
[GitHub link](https://github.com/qdrant/qdrant) ⭐ 29k

**Chroma**\
About: Open-source search and retrieval database for AI applications.\
[GitHub link](https://github.com/chroma-core/chroma) ⭐ 26.2k

**Pgvector**\
About: Open-source vector similarity search for Postgres.\
[GitHub link](https://github.com/chroma-core/chroma) ⭐ 19.9k

En este paper de la [Universidad de Cambridge](https://doi.org/10.1017/nlp.2024.53), dicen que ChromaDB es mejor que FAISS y Pinecone en términos generales de eficiencia y rendimiento.

En general, parece que Qdrant es más escalable y hecho para RAGs en producción con alta concurrencia, no en un entorno local en el que ChromaDB tendría más sentido. Y en cuanto a Milvus, tiene pinta de que es para proyectos mucho más ambiciosos, aunque no está nada mal.

---

## Parsing y limpieza
**Docling**\
Desarrollada por IBM y es capaz de trabajar con PDF, DOCX, PPTX, XLSX, HTML, WAV, MP3, WebVTT imágenes (PNG, TIFF, JPEG, ...), LaTeX y mucho más.\
[GitHub link](https://github.com/docling-project/docling) ⭐ 53.8k

**Marker**\
Es una herramienta altamente eficiente diseñada específicamente para convertir PDFs e imágenes a Markdown, JSON o HTML con gran precisión.\
[GitHub link](https://github.com/datalab-to/marker) ⭐ 31.8k

**Unstructured**\
Ofrece funciones directas (como partition_pdf()) que no solo devuelven texto, sino una lista de elementos (títulos, listas, texto narrativo) con metadatos como el número de página.\
[GitHub link](https://github.com/Unstructured-IO/unstructured) ⭐ 14k

**Nougat**\
Desarrollada por Meta. Es útil si necesitáramos extraer datos de papers, transforma esos papers a código markdown bastante bien.\
[GitHub link](https://github.com/facebookresearch/nougat) ⭐ 9.8k

---

## Estrategias de chunking
- **Chunking estructural**. Si se obtiene la información en markdown o json, se puede dividir el documento respetando su jerarquía (Encabezado 1, Encabezado 2, párrafos, bloques de código).

- **Chunking semántico**. Utilizar un modelo de embeddings para comparar frases consecutivas. Si la similitud matemática entre dos oraciones cae drásticamente, el sistema asume que el tema ha cambiado y hace un corte ahí.

- **Late chunking**. Invierte el orden tradicional. En lugar de cortar primero y vectorizar después, pasas el documento completo por un modelo de embeddings de contexto largo. El modelo procesa todo el texto, lo que permite que cada palabra comprenda su lugar en el documento a través del mecanismo de auto-atención (self-attention).

- **Contextual retrival**. Este enfoque utiliza modelos de lenguaje para generar un resumen o explicación del contexto global y se lo añade a cada pequeño fragmento antes de almacenarlo (a menudo combinado con técnicas de fusión de rangos como BM25 y re-ranking).

El este [paper](https://doi.org/10.1007/978-3-032-02899-0_1) se dice que para obtener la máxima precisión se debe usar contextual retrival, pero con un alto costo computacional como sacrificio (de VRAM sobre todo). No obstante, recomiendan un trade-off entre contextual retrival y late chunking.


---

## Modelos de embeddings


### Leaderboard ([Link](https://huggingface.co/spaces/mteb/leaderboard))

| Rank | Modelo | Score / Estado | Memoria (MB) | Parámetros (B) | Dimensiones | Max Tokens | Mean (Task) | Mean (Type) | Bitext Mining | Classification | Clustering | Instr. Retrieval | Multi. Class. | Pair Class. | Reranking | Retrieval | STS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | [KaLM-Embedding-Gemma3-12B-2511](https://kalm-embedding.github.io/) | 73% | 44884 | 11.76 | 3840 | 32768 | 72.32 | 62.51 | 83.76 | 77.88 | 55.77 | 5.49 | 33.03 | 84.73 | 67.27 | 75.66 | 79.02 |
| 2 | [llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b) | 99% | 28629 | 7.505 | 4096 | 32768 | 69.46 | 61.09 | 81.72 | 73.21 | 54.35 | 10.82 | 29.86 | 83.97 | 67.78 | 68.69 | 79.41 |
| 3 | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 99% | 14433 | 7.567 | 4096 | 32768 | 70.58 | 61.69 | 80.89 | 74 | 57.65 | 10.06 | 28.66 | 86.4 | 65.63 | 70.88 | 81.08 |
| 4 | [gemini-embedding-001](https://ai.google.dev/gemini-api/docs/embeddings) | 99% | null | null | 3072 | 2048 | 68.37 | 59.59 | 79.28 | 71.82 | 54.59 | 5.18 | 29.16 | 83.63 | 65.58 | 67.71 | 79.4 |
| 5 | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 99% | 7671 | 4.022 | 2560 | 32768 | 69.45 | 60.86 | 79.36 | 72.33 | 57.15 | 11.56 | 26.77 | 85.05 | 65.08 | 69.6 | 80.86 |
| 6 | [Seed1.6-embedding-1215](https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=doubao-embedding-vision) | 89% | null | null | 2048 | 32768 | 70.26 | 61.34 | 78.68 | 76.75 | 56.78 | -0.02 | 46.16 | 85.5 | 66.24 | 66.05 | 75.92 |
| 7 | [Octen-Embedding-8B](https://huggingface.co/bflhc/Octen-Embedding-8B) | 99% | 14433 | 7.567 | 4096 | 32768 | null | 60.18 | 80.35 | 66.68 | 55.68 | 8.9 | 25.23 | 85.12 | 67.64 | 70.77 | 81.27 |
| 8 | [jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small) | ⚠️ NA | 1137 | 0.596 | 1024 | 32768 | 67 | 58.9 | 69.71 | 71.32 | 53.41 | 1.35 | 41.97 | 82.93 | 65.66 | 64.88 | 78.85 |
| 9 | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 99% | 1136 | 0.596 | 1024 | 32768 | 64.34 | 56.01 | 72.23 | 66.83 | 52.33 | 5.09 | 24.59 | 80.83 | 61.41 | 64.65 | 76.17 |
| 10 | [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) | ⚠️ NA | 29040 | 7.069 | 3584 | 32768 | 62.51 | 55.93 | 73.92 | 61.55 | 52.77 | 4.94 | 25.48 | 85.13 | 65.55 | 60.08 | 73.98 |

---

## Prompting - Query transformation

- **Query reformulation**: a partir de la pregunta del usuario, un LLM la reformula antes de pasársela al LLM con el contexto.

- **Multi-Query Generation y RAG-Fusion**: a partir de la consulta original, le pides al LLM que la reescriba desde múltiples perspectivas generando varias versiones de la misma pregunta. Lanzas todas esas preguntas a la base de datos en paralelo y luego fusionas los resultados.

- **Step-back**: si el usuario hace una pregunta muy específica, el LLM primero genera una pregunta un paso atrás, es decir, mucho más abstracta o fundamental. El sistema busca información tanto para la pregunta específica como para la general, asegurando que el LLM final tenga todo el contexto base necesario para no perderse.

- **HyDE (Hypothetical Document Embeddings)**: en lugar de buscar directamente la pregunta del usuario, se usa un LLM para que genere una respuesta hipotética, aunque no sea 100% precisa. Luego, se vectoriza esa respuesta hipotética y la usas para buscar en tu base de datos.

- **Self-RAG**: utiliza el LLM para evaluar críticamente los documentos recuperados antes de generar la respuesta. Si el modelo determina que los documentos encontrados no responden a la pregunta, se autocorrige, por lo que puede reescribir la consulta automáticamente y volver a buscar.

En cuanto a este apartado, subí un [repositorio](https://github.com/athina-ai/rag-cookbooks/tree/main) interesante que encontré.


## LLMs

- GPT-4 / GPT4-4o / GPT4-4o-mini
- Claude 3.5 Sonnet
- Qwen2.5 7B, 14B o 32B Instruct
- Llama 3.1 8B
- Llama 3.2 3B
- Llama 3.3 70B estaría muy bien pero creo que ocupa +80GB de VRAM.