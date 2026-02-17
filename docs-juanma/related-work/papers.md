## From Questions to Insightful Answers: Building an Informed Chatbot for University Resources
[Paper link](https://arxiv.org/abs/2405.08120)

### Arquitectura
* **Pipeline RAG:** sistema denominado **BARKPLUG V.2** compuesto por dos fases: recuperación de contexto y completado (generación).
* **Retriever:** utiliza un modelo de embeddings de OpenAI (`text-embedding-3-large`) para transformar datos de 42 departamentos universitarios.
* **Base de datos vectorial:** los vectores se almacenan en **Chroma DB**, una base de datos en memoria.
* **Estrategia de búsqueda:** implementa una recuperación por umbral de similitud (Similarity Score Threshold) que devuelve resultados basados en un mínimo de confianza definido.
* **Generator:** utiliza el modelo **GPT-3.5-turbo** para generar respuestas personalizadas y fundamentadas factualmente.

### Evaluación
* **Cuantitativa (RAGAS):** obtuvo una puntuación media de **0.96**, demostrando alta precisión y relevancia en categorías como programas de ingeniería y recursos universitarios.
* **Métricas End-to-End:** se evaluó la similitud de respuesta (aprox. 0.83) y la corrección de respuesta (aprox. 0.88) para asegurar la fidelidad de los datos.
* **Usabilidad (SUS):** una evaluación con 50 estudiantes mediante la *System Usability Scale* resultó en una puntuación de **67.75**, validando una experiencia de usuario satisfactoria.

---

## Design and Performance Evaluation of LLM-Based RAG Pipelines for Chatbot Services in International Student Admissions
[Paper link](https://www.mdpi.com/2079-9292/14/15/3095)

### Arquitectura
* **Configuraciones de pipeline:** experimentación con múltiples combinaciones de fragmentación (Recursive vs. Semantic) y métodos de recuperación.
* **Retriever:** comparativa entre métodos **MMR**, **Dense**, e **Híbrido** (BM25 + Dense), además de técnicas avanzadas como **MultiQuery** y **MultiVector**.
* **Modelos (LLMs):** evaluación de modelos comerciales (**GPT-4o**) frente a modelos de código abierto (**LLaMA3**, **OpenChat**, **Zephyr**, **Neural-chat**) ejecutados localmente con Ollama.
* **Embeddings:** uso de `all-MiniLM-L6-v2` (384 dimensiones) con almacenamiento en un índice **FAISS** basado en similitud de coseno.

### Evaluación
* **Dataset dual:** evaluación basada en un conjunto de QA generado por LLM (72 pares) y otro **etiquetado por humanos** (62 pares) en colaboración con estudiantes internacionales.
* **Métricas de latencia:** el modelo LLaMA3 local mostró ser significativamente más rápido (1.58s) que GPT-4o (3.03s) en la configuración más veloz.
* **Optimización open source:** se demostró que el modelo **OpenChat-7b** (v3.5-0106-fp16) con recuperación MultiQuery Híbrida puede superar a GPT-4o en precisión (promedio RAGAS de **0.7377** vs **0.7249**).

---

## Development and Evaluation of a University Chatbot Using Deep Learning: A RAG-Based Approach
[Paper link](https://link.springer.com/chapter/10.1007/978-3-031-88045-2_7)

### Arquitectura
* **Componentes principales:** orquestación con **LangChain**, almacenamiento de conocimiento externo en un vector store y una interfaz de usuario en **Streamlit**.
* **Embeddings:** implementación del modelo `text-embedding-ada-002` de OpenAI por su alta precisión semántica.
* **Fragmentación (chunking):** división del texto en fragmentos de **1000 caracteres con un traslape de 200** para optimizar la eficiencia computacional y preservar el contexto.
* **Recuperación:** uso de **FAISS** para realizar búsquedas semánticas mediante similitud de coseno ante las consultas del usuario.

### Evaluación
* **Comparativa de modelos:** evaluación del sistema RAG utilizando GPT-4, GPT-4 mini y GPT-3.5.
* **Precisión humana:** en una prueba de 30 preguntas, el sistema RAG logró un **90% de respuestas correctas**, superando ampliamente a GPT-4 independiente (66.6%) y GPT-3.5 (0% en datos específicos).
* **RAGAS:** el sistema alcanzó puntuaciones de entre **0.8 y 0.9** en precisión de contexto, fidelidad, relevancia de respuesta y corrección.

---

## Reinforcement Learning for Optimizing RAG for Domain Chatbots
[Paper link](https://arxiv.org/abs/2401.06800)

### Arquitectura
* **Agente de Política RL:** un modelo externo (**BERT** o **GPT-2**) que decide dinámicamente si ejecutar la acción `[FETCH]` (recuperar contexto del FAQ) o `[NO_FETCH]` (omitir para ahorrar tokens).
* **Embedding In-house:** modelo entrenado con pérdida **infoNCE** para maximizar la similitud entre consulta y respuesta, superando a modelos públicos generales.
* **Estado:** definido por las consultas previas, las acciones previas de la política y la consulta actual.
* **Generador:** uso de la API de OpenAI para el modelo **GPT-3.5-turbo** (16k context).

### Evaluación
* **Modelo de recompensa:** se utilizó **GPT-4** para calificar las respuestas del bot como "Good" o "Bad", convirtiendo esto en una recompensa numérica para entrenar el modelo de política.
* **Ahorro de costos:** la optimización basada en RL junto con un umbral de similitud logró reducir el uso de tokens en un **31%**.
* **Precisión de recuperación:** el modelo entrenado con infoNCE alcanzó una precisión **top-1 del 97%** en consultas en inglés y **94%** en Hinglish, superando notablemente al modelo base e5-base-v2.