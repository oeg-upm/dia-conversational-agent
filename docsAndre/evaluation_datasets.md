**Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets**
https://doi.org/10.48550/arXiv.2504.20119

Analiza cómo se están evaluando actualmente los sistemas de **Retrieval-Augmented Generation (RAG)** y si los propios LLMs pueden utilizarse como evaluadores automáticos. 
El paper destaca que las métricas clásicas no son suficientes y propone separar la evaluación en **retrieval**, **generación** y **pipeline completo**, utilizando métricas como *Recall@k*, *faithfulness* y *answer relevance*. 
También subraya la importancia de usar **datasets específicos del dominio** en lugar de benchmarks genéricos. 

En resumen, diseñar datasets propios basados en las learning guides, evaluar por separado recuperación y generación, y usar herramientas automáticas (como evaluadores LLM o métricas tipo RAGAS) complementadas con revisión humana para analizar desempeño, seguridad y confiabilidad del sistema.

Tomando en cuenta que nuestro corpus es muy espeífico tal vez otros datasets no sean tan eficientes como uno hecho por nosotros, pero tampoco lo haremos manualmente. Tiene pocas estrellas pero usaría:https://github.com/OpenBMB/RAGEval/tree/c44aa98aea18362eb5e3751f11e13dbc8f6cab83/rageval/qar_generation

Alternativamente y tomando en cuenta **Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems**
(Teixeira de Lima et al., COLING 2025) https://aclanthology.org/2025.coling-industry.4/ podemos seguir una metodología hibrída entre un LLM y supervisión humana.

**Dataset Generation Strategy for Evaluating the RAG System**

Para evaluar el sistema RAG construiremos un dataset específico basado en las **learning guides**, siguiendo la taxonomía propuesta del paper. El objetivo es medir distintas capacidades del sistema como recuperación de información, razonamiento sobre múltiples fragmentos y manejo de preguntas sin respuesta.

La generación del dataset seguirá una **estrategia híbrida**. Primero se seleccionará un subconjunto representativo de learning guides y cada documento será dividido en fragmentos utilizando el mismo método de *chunking* del sistema RAG. A partir de estos fragmentos se generarán preguntas automáticamente con un LLM, solicitando preguntas que puedan responderse usando únicamente el contenido del texto. Posteriormente se generarán también preguntas que requieran combinar múltiples fragmentos y preguntas cuya respuesta no esté en el corpus. Finalmente, todas las preguntas serán revisadas manualmente para asegurar claridad, calidad y correcta correspondencia con el tipo de evaluación deseado.

| Tipo de pregunta | Objetivo de evaluación | # Preguntas |
|---|---|---|
| Single-hop retrieval | Evaluar si el sistema recupera correctamente un fragmento con la respuesta directa | 40 |
| Multi-hop reasoning | Evaluar si el sistema puede combinar información de múltiples fragmentos | 25 |
| Aggregation | Evaluar la capacidad de sintetizar información distribuida en el documento | 15 |
| Unanswerable questions | Evaluar si el sistema reconoce cuando la respuesta no está en el corpus | 20 |
| Adversarial questions | Evaluar robustez frente a preguntas ambiguas o diseñadas para provocar errores | 15 |


