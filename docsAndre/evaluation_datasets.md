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

# Safety & Compliance Evaluation V2

Este documento describe la metodología y el marco teórico detrás de la evaluación de seguridad y cumplimiento del asistente conversacional basado en RAG desarrollado para el Departamento de IA de la universidad.

> **Note:** Esta evaluación se desarrolla en paralelo con la evaluación de rendimiento técnico realizada por otros miembros del equipo — la cual mide la calidad de recuperación, el razonamiento y la fidelidad al corpus. Ambas evaluaciones son ortogonales y necesarias: un sistema puede recuperar y generar correctamente y aun así fallar en sus obligaciones de seguridad, transparencia y uso adecuado.

---

## Overview

La pregunta central que responde esta evaluación no es *"¿el sistema responde bien?"* sino *"¿el sistema se comporta como debería?"*

El sistema es un asistente RAG grounded: está diseñado para responder preguntas exclusivamente basándose en el contenido de cientos de guías docentes (documentos PDF) de los programas de grado y máster de la universidad. No debe responder a preguntas cuyas respuestas no estén en el corpus, no debe ser utilizado para fines fuera de su alcance, y debe cumplir con las obligaciones de transparencia establecidas por el EU AI Act para sistemas de IA interactivos.

Para evaluar estas propiedades de forma empírica, construimos un dataset de **60 prompts adversariales** que analizan el comportamiento del sistema a lo largo de tres dimensiones derivadas del marco de confiabilidad propuesto por Zhou et al. (2024).

---

## Theoretical Framework

### Trustworthiness in RAG Systems — Zhou et al. (2024)

La base teórica de esta evaluación es el marco unificado de confiabilidad propuesto por Zhou et al. en *Trustworthiness in Retrieval-Augmented Generation Systems: A Survey* (arXiv:2409.10102). El marco define seis dimensiones para evaluar la confiabilidad de un sistema RAG: **factuality, robustness, fairness, transparency, accountability, and privacy**.

Este marco hace explícito que la evaluación de rendimiento por sí sola es insuficiente: un sistema RAG puede ser técnicamente preciso y aun así producir contenido inapropiado, carecer de transparencia o ser vulnerable a manipulaciones.

De las seis dimensiones:
- **Factuality** ya está cubierta por el dataset de evaluación de rendimiento del equipo.
- **Privacy** y **accountability** no pueden evaluarse mediante prompts en nuestro sistema, ya que las guías docentes no contienen datos personales y los mecanismos de gobernanza se validan a través de la documentación del proyecto.
- **Robustness** y **transparency** pueden evaluarse empíricamente mediante el comportamiento observable del sistema — y constituyen la base de este dataset.

Una tercera dimensión no cubierta explícitamente por Zhou et al., pero crítica para un chatbot de ámbito institucional, es el **appropriate use**: la capacidad del sistema para rechazar preguntas que están fuera de su propósito, son ofensivas o están diseñadas para manipular su comportamiento. Esta dimensión se fundamenta en el Article 52 del EU AI Act y en la distinción entre *epistemic refusal* y *policy refusal* formalizada por Muhamed et al. (2025).

---

### The Two Forms of Refusal — Muhamed et al., NeurIPS 2025

Un eje conceptual clave que organiza esta evaluación es la distinción entre dos tipos de rechazo que un asistente RAG institucional debe ser capaz de ejecutar, introducida por Muhamed et al. en *RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models* (NeurIPS 2025).

**Epistemic refusal** ocurre cuando el sistema no puede responder porque la información requerida no está en el corpus. La pregunta puede ser completamente legítima, pero la respuesta simplemente no está disponible en las guías docentes. Muhamed et al. demuestran que esta capacidad es un punto de fallo sistemático incluso en modelos avanzados, con una precisión de rechazo inferior al 50% en tareas multi-documento, y que implica dos habilidades separables — detectar la insuficiencia y categorizar su tipo — que no mejoran con el tamaño del modelo.

**Policy refusal** ocurre cuando el sistema debe rechazar responder independientemente de si la información existe, porque la pregunta es inapropiada, ofensiva, está fuera de alcance o busca manipular el comportamiento del sistema. Esta dimensión está directamente vinculada a las obligaciones del EU AI Act y a la protección de la reputación institucional de la universidad.

---

### Dataset Construction Methodology

El dataset fue construido siguiendo dos principios metodológicos de la literatura:

**Automated red teaming — Perez et al., EMNLP 2022**

En lugar de escribir casos de prueba manualmente — lo cual es costoso y limita la diversidad — utilizamos un LLM para generarlos automáticamente a partir de instrucciones que apuntan a modos de fallo específicos. Este es el principio central del *automated red teaming* propuesto por Perez et al. en *Red Teaming Language Models with Language Models* (EMNLP 2022), que demuestra que los casos de prueba generados por modelos pueden descubrir comportamientos dañinos sistemáticos a gran escala y con diversidad controlada.

En la práctica, esto se implementa en [`safety_prompt_generator.py`](./safety_prompt_generator.py), que utiliza un LLM local a través de la API de Groq para generar prompts adversariales por categoría, guiado por instrucciones específicas que definen el tipo de fallo a provocar y la estrategia de disfraz académico a utilizar.

**Perturbation of existing dataset questions — SafeRAG, Liang et al., ACL 2025**

Parte de los casos de prueba se construyen tomando preguntas respondibles del dataset de evaluación de rendimiento del equipo — preguntas reales sobre contenido de las guías docentes — y modificándolas de forma controlada para introducir premisas falsas, información parcialmente incorrecta o instrucciones ocultas.

Este enfoque está inspirado en SafeRAG (*Benchmarking Security in Retrieval-Augmented Generation of Large Language Model*, ACL 2025), que demuestra que los casos adversariales más efectivos y realistas son precisamente aquellos disfrazados como consultas legítimas, ya que reflejan el tipo de interacción que un usuario real podría generar, de forma intencional o no. La metodología de SafeRAG, basada en la perturbación sistemática de documentos base en lugar de la generación de contenido completamente nuevo, maximiza el realismo del dataset y asegura que los fallos representen patrones de uso reales.

La combinación de ambos enfoques produce un dataset que cubre tanto escenarios adversariales completamente nuevos como variantes adversariales de interacciones reales del sistema.

---

## Dataset Taxonomy

El dataset está organizado en **tres dimensiones** derivadas del marco de Zhou et al. (2024), operacionalizadas mediante los conceptos de Muhamed et al. (2025), Perez et al. (2022) y el EU AI Act.

---

### Dimension 1 — Epistemic Robustness

> *Framework: Zhou et al. (2024), Robustness dimension; Muhamed et al. (2025), epistemic refusal*

Evalúa si el sistema reconoce cuándo no puede responder correctamente y se abstiene en lugar de fabricar una respuesta. Se divide en dos sub-dimensiones correspondientes a las principales causas de fallo epistémico identificadas por RefusalBench:

**1.1 Missing information**

Preguntas plausibles y relevantes cuyo contenido no está presente en ninguna guía docente del corpus. El sistema de recuperación puede devolver un documento semánticamente similar que no contiene la respuesta específica. El sistema debe reconocer esta insuficiencia y rechazar responder en lugar de alucinar.

**1.2 Defective context**

Preguntas construidas mediante la perturbación de elementos del dataset de evaluación de rendimiento, introduciendo premisas falsas, datos contradictorios o desajustes epistémicos — por ejemplo, preguntar por una nota mínima incorrecta o por una política que nunca existió pero suena institucional. El sistema debe detectar que el contexto recuperado no respalda la premisa de la pregunta.

---

### Dimension 2 — Robustness Against Inappropriate Use

> *Framework: Zhou et al. (2024), Robustness dimension; Perez et al. (2022), red teaming; Liang et al. (2025), adversarial generation*

Evalúa si el sistema mantiene su comportamiento esperado ante intentos de manipulación o uso fuera de su propósito. Se divide en dos sub-dimensiones:

**2.1 Prompt injection**

Preguntas que contienen instrucciones ocultas disfrazadas como consultas académicas, generadas automáticamente mediante la metodología de red teaming de Perez et al. (2022), con variaciones sistemáticas: role-play, ataques de continuación de instrucciones, etiquetas falsas de sistema, cambios de idioma a mitad de la consulta y formulaciones hipotéticas. El sistema debe ignorar estas instrucciones y comportarse conforme a sus restricciones normales.

**2.2 Policy refusal**

Preguntas inapropiadas, ofensivas, sesgadas o completamente fuera del propósito del sistema — incluso si están redactadas en lenguaje académico. Esto incluye temas sensibles, solicitudes de opinión indebidas y preguntas fuera del dominio de las guías docentes. El sistema debe rechazar responder independientemente de su capacidad técnica para hacerlo. Esta sub-dimensión evalúa que el prompt del sistema actúe como una barrera efectiva frente a usos no previstos.

---

### Dimension 3 — Transparency

> *Framework: Zhou et al. (2024), Transparency dimension; EU AI Act, Article 52*

Evalúa el cumplimiento de las obligaciones de identificación y divulgación establecidas por el EU AI Act para sistemas de IA interactivos. El sistema debe:

- Identificarse explícitamente como un sistema de IA cuando se le pregunte directa o indirectamente  
- Indicar que sus respuestas son generadas automáticamente  
- Evitar presentarse como una autoridad institucional o como el profesor del curso  

---

## Dataset Distribution

| Dimension | Sub-dimension | Theoretical basis | # Prompts |
|---|---|---|---|
| Epistemic robustness | Missing information | RefusalBench — Muhamed et al., NeurIPS 2025 | 15 |
| Epistemic robustness | Defective context | RefusalBench + SafeRAG — Liang et al., ACL 2025 | 10 |
| Robustness against inappropriate use | Prompt injection | Perez et al., EMNLP 2022 + SafeRAG | 15 |
| Robustness against inappropriate use | Policy refusal | Perez et al., EMNLP 2022 | 10 |
| Transparency | Identification and disclosure | AI Act Art. 52 + Zhou et al., 2024 | 10 |
| **Total** | | | **60** |

---

## What Cannot Be Evaluated Through This Dataset

Algunos criterios de cumplimiento corresponden a aspectos organizativos o de infraestructura y no pueden verificarse mediante prompts.

**Personal data protection (GDPR):** No aplicable, ya que las guías docentes no contienen datos personales ni información sensible de estudiantes o profesores.

**Infrastructure security:** El control de acceso a la API, la protección de la base de datos vectorial y el cifrado en tránsito dependen de la arquitectura del sistema, no del comportamiento del modelo ante inputs.

**System governance:** La asignación de responsables, la clasificación de riesgos y la documentación técnica requerida por el AI Act se validan mediante la documentación del proyecto, no mediante evaluación empírica del modelo.

---

## References

- Zhou, Y., et al. (2024). *Trustworthiness in Retrieval-Augmented Generation Systems: A Survey*. arXiv:2409.10102.
- Muhamed, A., et al. (2025). *RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models*. NeurIPS 2025.
- Liang, X., et al. (2025). *SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model*. ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.230
- Perez, E., et al. (2022). *Red Teaming Language Models with Language Models*. EMNLP 2022. https://doi.org/10.48550/arXiv.2202.03286
- European Parliament and Council. (2024). *Regulation (EU) 2024/1689 (AI Act)*, Article 52.


