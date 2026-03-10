## RAG Evaluation

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

---

## Safety and Compliance Evaluation

Además de evaluar el desempeño técnico del sistema RAG, también es importante analizar su comportamiento desde una perspectiva de **seguridad y mitigación de riesgos**. Aunque el sistema no está clasificado como de alto riesgo bajo el AI Act, varios marcos recientes proponen evaluar el comportamiento de los LLM frente a distintos tipos de riesgos asociados a su uso interactivo.


**Holistic Evaluation of Language Models (Liang et al., 2023)**  
https://doi.org/10.48550/arXiv.2211.09110

Este trabajo introduce HELM, un framework que propone evaluar los modelos de lenguaje en múltiples dimensiones además de la precisión, incluyendo **robustez, confiabilidad, seguridad y transparencia**. En lugar de evaluar únicamente la exactitud de las respuestas, el framework propone analizar cómo se comporta el sistema frente a diferentes tipos de prompts y escenarios.

Siguiendo esta línea, en este proyecto evaluaremos el comportamiento del sistema RAG ante distintos tipos de interacciones que puedan revelar riesgos asociados al uso de LLM.

**Red Teaming Language Models with Language Models (Perez et al., 2022)**  
https://doi.org/10.48550/arXiv.2202.03286

Este trabajo introduce el concepto de **red teaming para LLM**, en el que se diseñan prompts adversariales con el objetivo de identificar comportamientos inesperados o inseguros del sistema.

Siguiendo este enfoque, incluiremos prompts diseñados para intentar manipular el comportamiento del sistema, por ejemplo intentando que el modelo ignore las restricciones del RAG o que genere respuestas basadas en conocimiento externo.

**Qué partes de la matriz pueden evaluarse experimentalmente**

A partir de la matriz de compliance del proyecto, algunos criterios pueden evaluarse directamente mediante el comportamiento observable del sistema.

**Transparencia del sistema**

El AI Act establece que los usuarios deben ser informados cuando interactúan con un sistema de IA y cuando el contenido ha sido generado artificialmente.

Esto se evaluará verificando si el sistema:

- se identifica explícitamente como sistema de IA
- indica que las respuestas son generadas automáticamente
- evita presentarse como una autoridad institucional

**Robustez frente a alucinaciones**

Siguiendo las recomendaciones del survey de Survey of Hallucination in Natural Language Generation Ji et al. (2023), se evaluará si el sistema genera información incorrecta cuando no dispone de contexto suficiente.

Esto se realizará mediante preguntas cuya respuesta **no se encuentra en el corpus de learning guides**, verificando si el sistema:

- reconoce que no puede responder
- evita generar información no respaldada por los documentos

**Robustez frente a prompts adversariales**

Siguiendo la metodología de **red teaming** propuesta por Perez et al. (2022), se evaluará si el sistema es vulnerable a prompts diseñados para manipular su comportamiento, por ejemplo intentando que ignore las instrucciones del sistema o que utilice conocimiento externo.

---

**Qué partes de la matriz no pueden evaluarse mediante el dataset**

Algunos criterios de la matriz corresponden a aspectos **organizativos o de infraestructura**, por lo que no pueden evaluarse mediante prompts o datasets.

Protección de datos personales (GDPR)

Los criterios relacionados con tratamiento de datos personales no pueden evaluarse mediante este dataset, ya que las **learning guides no contienen información personal ni datos sensibles**.

Seguridad de infraestructura

Aspectos como control de acceso a la API, seguridad de la base vectorial o protección de la infraestructura dependen de la arquitectura del sistema y no del comportamiento del modelo frente a prompts.

Gobernanza del sistema

Elementos como la designación de responsables del sistema, la clasificación de riesgo o la documentación técnica forman parte de la gobernanza del sistema y se validan mediante documentación del proyecto.

---

**Dataset de evaluación de seguridad y compliance**

Siguiendo las metodologías de **HELM (Liang et al., 2023)** y **red teaming (Perez et al., 2022)**, se puede construir un dataset de prompts diseñado para evaluar el comportamiento del sistema frente a distintos tipos de riesgos.

| Tipo de prueba | Objetivo de evaluación | # Prompts |
|---|---|---|
| Transparency checks | Verificar que el sistema se identifique como IA y no como autoridad institucional | 10 |
| AI-generated content disclosure | Verificar que las respuestas indiquen que son generadas por IA | 10 |
| Uncertainty handling | Evaluar si el sistema reconoce cuando no tiene suficiente información | 10 |
| Prompt injection attempts | Evaluar robustez frente a prompts adversariales | 15 |
| Hallucination resistance | Evaluar si el sistema evita generar información no respaldada por el corpus | 15 |

En total, el dataset de seguridad y compliance estará compuesto por **60 prompts**.


