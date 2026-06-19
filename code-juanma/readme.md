# Arquitectura del Agente Conversacional RAG

Este directorio contiene el núcleo del desarrollo técnico, los algoritmos y los marcos de evaluación implementados para el Trabajo Fin de Máster titulado **"Agente Conversacional para el Departamento de Inteligencia Artificial"**, realizado por **Juan Manuel Cardeñosa Borrego** en la Universidad Politécnica de Madrid (UPM).

El software aquí alojado abarca desde las primeras pruebas conceptuales de ingesta documental hasta el pipeline de producción final, incluyendo los entornos automatizados para auditar de forma empírica la precisión factual de los modelos y su comportamiento seguro.

## Estructura del proyecto

El código está organizado de forma modular en las siguientes subcarpetas, reflejando el ciclo de vida y la evolución iterativa de la investigación:

* **`basic-code/`**: contiene la primera versión beta y el prototipo conceptual del pipeline RAG. Sirvió como entorno inicial de pruebas para validar los mecanismos básicos de extracción de texto, vectorización elemental y comunicación con las APIs de los modelos de lenguaje locales.
* **`dataset/`**: módulo dedicado a la ingesta avanzada de las guías de aprendizaje universitarias mediante la herramienta *Docling* (IBM). Incluye los scripts de generación sintética que estructuran el banco de pruebas bajo una taxonomía formal de cinco categorías de consulta (`factual`, `procedural`, `comparative`, `out_of_scope` y `ambiguous`).
* **`evaluation/`**: entorno de ejecución y auditoría algorítmica basado en el enfoque *LLM-as-a-judge*. Contiene los scripts encargados de lanzar las baterías de pruebas iterativas y computar las métricas clave del sistema (*Context Precision*, *Faithfulness*, *Answer Correctness* y *Safe Behavior*), guardando los resultados empíricos que sustentan las conclusiones del TFM.
* **`rag-code/`**: repositorio con el código fuente del sistema RAG definitivo. Implementa la arquitectura optimizada (*Naive RAG* combinado con modelos de embeddings de alta dimensionalidad como `octen-4b`), las directivas de control y la lógica del asistente conversacional robusto para entornos institucionales.

## Flujo técnico recomendado

Para comprender la ejecución y replicar los experimentos presentados en la memoria académica, se sugiere seguir el siguiente orden lógico:

1. **Exploración inicial (`basic-code/`)**: revisión de los fundamentos y del diseño base del conducto de recuperación.
2. **Preparación documental (`dataset/`)**: ejecución de los parses estructurales y carga de los datasets sintéticos validados.
3. **Despliegue del sistema (`rag-code/`)**: inicialización del pipeline principal del agente sobre los servidores de inferencia locales.
4. **Auditoría e inferencia (`evaluation/`)**: lanzamiento de los scripts comparativos para medir el impacto de las diferentes hipótesis arquitectónicas (H1a, H1b, H1c y H2).

---
*Repositorio oficial del Trabajo Fin de Máster de Juan Manuel Cardeñosa Borrego - Inteligencia Artificial, ETSI Informáticos, UPM (2026).*