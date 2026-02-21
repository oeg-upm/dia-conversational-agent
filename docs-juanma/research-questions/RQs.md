## A Systematic Literature Review of Retrieval-Augmented Generation: Techniques, Metrics, and Challenges

[Paper link](https://doi.org/10.3390/bdcc9120320)

**RQ1: ¿Qué métodos y enfoques innovadores amplían el marco estándar de RAG?**\
Entre las principales innovaciones destacan la recuperación híbrida y consciente de la estructura (como el uso de grafos), los bucles de recuperación iterativos que se activan ante la incertidumbre del modelo, la integración de memoria, la orquestación mediante agentes y la adopción de la multimodalidad.


**RQ2: ¿Qué métricas se utilizan con mayor frecuencia para evaluar la efectividad de los sistemas RAG?**\
Aunque siguen predominando las métricas clásicas basadas en la superposición de texto (como Exact Match, F1, BLEU y ROUGE), el campo exige cada vez más el uso de métricas de diagnóstico de recuperación (ej. Recall@k, MRR@k, nDCG). Además, para evaluar la fidelidad de las respuestas, se ha vuelto indispensable el uso de evaluaciones humanas y protocolos donde un modelo de lenguaje actúa como juez (LLM-as-judge).

**RQ3: ¿Qué desafíos y limitaciones están asociados con las técnicas RAG?**\
Los retos persistentes se agrupan en cinco áreas principales: la calidad de la evidencia (datos ruidosos o desactualizados), los errores en cascada a través de las arquitecturas modulares y el coste de los recursos (presupuestos de latencia y tokens). Por último, preocupan las restricciones propias de los LLMs (como su límite de contexto y alucinaciones) y las crecientes amenazas a la seguridad, incluyendo el envenenamiento de datos, la filtración de información privada y la inyección de prompts.

---

## QuIM-RAG: Advancing Retrieval-Augmented Generation With Inverted Question Matching for Enhanced QA Performance

[Paper link](https://ieeexplore.ieee.org/abstract/document/10781379)

**RQ1: ¿Cómo se desempeña un modelo RAG avanzado con un mecanismo de recuperación novedoso en comparación con un modelo RAG convencional?**\
Su mecanismo de recupearación (QuIM-RAG) convierte la búsqueda de la respuesta en un proceso de coincidencia entre la pregunta del usuario y una serie de preguntas potenciales generadas previamente para cada fragmento de texto.
Destacan una mayor fidelidad, recuperación y precisión del contexto, mayor calidad semántica y reducción de errores.

---

## RAG based Question-Answering for Contextual Response Prediction System

[Paper link](https://doi.org/10.48550/arXiv.2409.03708)

**RQ1: ¿Cuáles son los efectos de diferentes técnicas de embeddings, estrategias de recuperación y métodos de prompting en el rendimiento de RAG?**\
La combinación de embeddings de Vertex AI con la recuperación ScaNN fue la más precisa y eficiente. Por el contrario, los métodos de prompting complejos (como CoVe o CoTP) resultaron ser muy lentos y redujeron la precisión, por lo que fueron descartados.

**RQ2: ¿Puede el prompting ReAct (Reason+Act) mejorar la precisión factual y reducir las alucinaciones de los LLM en entornos en tiempo real?**\
Aunque el uso de ReAct logró reducir las alucinaciones y mejorar la precisión, hizo que el sistema fuera demasiado lento, volviéndolo poco práctico para su uso en atención al cliente en tiempo real.

---

### Otras posibles preguntas (no respaldadas por papers):
**RQ1: ¿Cómo afectan las diferentes estrategias de chunking a las métricas de recuperación bajo distintas configuraciones?**

**RQ2: ¿En qué medida varía la tasa de alucinaciones y la fidelidad al contexto al cambiar el LLM manteniendo constante el contexto recuperado?**

**RQ3: ¿Cómo afecta la densidad de información en el contexto recuperado (número de chunks inyectados en el prompt) a la calidad de la generación y a la latencia de inferencia?**

**RQ4: ¿Qué grado de correlación existe entre las evaluaciones automáticas realizadas por RAGAS y las evaluaciones manuales?**