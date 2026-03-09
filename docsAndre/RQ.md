# Línea A — Evaluación del sistema RAG

## Pregunta A1

**¿Cómo afectan diferentes configuraciones de RAG a la calidad de las respuestas generadas por un modelo compacto que opera sobre un corpus limitado de documentos (learning guides)?**

Tomando en cuenta las iniciales limitaciones (en llms y poder computacional) debemos entender qué tanto puede mejorar el desempeño del sistema al aplicar distintas configuraciones de RAG (chunking, embeddings, estrategias de retrieval, etc.).

Nuestro corpus contiene más de 1000 documentos pero todos con las mismas 8 secciones, por lo que necesitamos analizar si el sistema es capaz de recuperar correctamente la información relevante dentro de este tipo de documentos y generar respuestas útiles.

**Paper**

Retrieval-Augmented Generation for Large Language Models: A Survey  
Gao et al., 2023  
https://doi.org/10.48550/arXiv.2312.10997

---

## Pregunta A2

**¿Qué tan efectiva es la etapa de recuperación de información cuando se trabaja con un corpus documental estructurado como las learning guides?**

Dado que estamos utilizando modelos compactos, la calidad del sistema depende en gran medida de que el retrieval encuentre los fragmentos correctos del documento antes de generar una respuesta. Si el sistema recupera información incorrecta o irrelevante, el modelo probablemente generará respuestas equivocadas.

En nuestro caso los documentos tienen una estructura relativamente consistente (las mismas secciones en cada learning guide), por lo que queremos analizar si el sistema es capaz de aprovechar esta estructura para mejorar la recuperación de información.

**Paper**

ARES: Automated Evaluation of Retrieval-Augmented Generation Systems  
Saad-Falcon et al., 2023  
https://doi.org/10.48550/arXiv.2311.09476

---

## Pregunta A3

**¿En qué medida el uso de RAG permite reducir las alucinaciones cuando se utilizan modelos compactos ejecutados localmente?**

Los modelos de lenguaje pequeños o medianos pueden ser más propensos a generar respuestas incorrectas o inventadas cuando no tienen suficiente contexto. Una de las motivaciones principales de usar RAG es reducir este problema obligando al modelo a basarse en documentos reales.


**Paper**

RAGAS: Automated Evaluation of Retrieval-Augmented Generation  
Es et al., 2024 (EACL)  
https://doi.org/10.18653/v1/2024.eacl-demo.16

---

## Pregunta A4

**¿Qué tan robusto es el sistema cuando recibe preguntas complejas o preguntas que no pueden responderse con los documentos disponibles?**

En un entorno real, los usuarios pueden hacer preguntas que no necesariamente tienen respuesta dentro de los documentos disponibles. Dado que nuestro sistema inicialmente solo tiene acceso a las learning guides, es importante evaluar cómo se comporta cuando el usuario hace preguntas fuera del alcance del corpus.

Queremos analizar si el sistema reconoce correctamente estos casos o si intenta generar respuestas inventadas a partir de información incompleta.

**Paper**

RAGBench: Benchmarking Retrieval-Augmented Generation Systems  
2024  
https://doi.org/10.48550/arXiv.2407.11005

---

# Línea B — Seguridad y Compliance

## Pregunta B1

**¿Qué tan vulnerable es el agente conversacional a ataques de prompt injection cuando trabaja sobre un sistema RAG basado en documentos?**

Los sistemas basados en LLM pueden ser manipulados mediante prompts diseñados para alterar su comportamiento. Por ejemplo, un usuario podría intentar convencer al modelo de ignorar las instrucciones del sistema o responder sin utilizar los documentos disponibles.

En nuestro caso queremos evaluar si el agente conversacional puede ser manipulado para generar respuestas que no estén basadas en las learning guides o que ignoren las restricciones del sistema.

**Paper**

HarmBench: A Standardized Evaluation Framework for LLM Safety  
Mazeika et al., 2024  
https://doi.org/10.48550/arXiv.2402.04249

---

## Pregunta B2

**¿Las respuestas generadas por el sistema mantienen trazabilidad hacia los documentos fuente utilizados?**

Para que un sistema conversacional basado en documentación institucional sea confiable, es importante que las respuestas puedan ser verificadas y rastreadas hasta los documentos originales.

En este proyecto queremos evaluar si el sistema es capaz de indicar de qué documento o fragmento proviene la información utilizada para responder, lo cual es especialmente importante cuando se trabaja con más de mil learning guides.

**Paper** 
TruthfulQA: Measuring How Models Mimic Human Falsehoods  
Lin et al., ACL 2022  
https://doi.org/10.1162/tacl_a_00415

Tanto este como la A4 pueden tomar las guidelines para crear los datasets que impliquen estos puntos:

preguntas que sí están en las learning guides

preguntas que requieren juntar 2 secciones

preguntas ambiguas

preguntas que no están en las learning guides

prompts que intentan hacer que el sistema responda fuera del corpus

---

## Pregunta B3

**¿Cómo responde el sistema ante preguntas relacionadas con información sensible o potencialmente personal?**

Aunque el sistema está diseñado para trabajar principalmente con documentos institucionales, los usuarios podrían hacer preguntas relacionadas con información sensible o datos personales.

Desde una perspectiva de seguridad y cumplimiento, queremos analizar si el agente responde de forma responsable ante este tipo de preguntas y evita generar información inapropiada o potencialmente problemática.

**Paper** (No he encontrado uno mejor pero este va dirigido a contenido sensible u ofensivo)

BeaverTails: Towards Improved Safety Alignment of LLMs  
Ji et al., 2023  
https://doi.org/10.48550/arXiv.2307.04657
