# Generación del dataset de evaluación

## Taxonomía del dataset

El conjunto de datos se compone de cinco categorías de consulta bien definidas:

1. **Consultas factuales (`factual`):** preguntas unívocas y directas sobre datos concretos contenidos en las guías (ej. *¿Cuántos créditos ECTS tiene la asignatura X?* o *¿Quién es el coordinador?*).
2. **Consultas procedimentales (`procedural`):** preguntas que exigen la descripción de pasos secuenciales o trámites administrativos (ej. *¿Cuál es el proceso para inscribirme en la asignatura Y?*).
3. **Consultas comparativas (`comparative`):** evaluaciones cruzadas que obligan al sistema a contrastar información de múltiples bloques o asignaturas (ej. *Comparar los criterios de evaluación entre la asignatura A y la asignatura B*).
4. **Consultas fuera de dominio (`out_of_scope`):** preguntas no relacionadas con el contexto de las guías docentes para evaluar la capacidad de contención del modelo (ej. *¿Cuál es la capital de Francia?*). El sistema debe rechazar responder usando su conocimiento interno.
5. **Consultas ambiguas (`ambiguous`):** preguntas formuladas de manera incompleta o vaga para verificar si el agente solicita aclaraciones en lugar de alucinar o asumir datos erróneos.


## Estructura del directorio y componentes

A continuación se detalla la organización de los archivos, scripts y conjuntos de datos incluidos en esta ruta, los cuales reflejan la evolución técnica e iterativa del proceso de generación y evaluación del TFM:

### Carpetas y datos
* **`datasets/`**: carpeta que almacena los conjuntos de datos de prueba y entrenamiento generados durante las fases iniciales (versiones V1 y V2) para validar el comportamiento del motor de ingesta.
* **`evaluation_results_V1.csv` y `evaluation_results_V2.csv`**: archivos de resultados en formato estructurado que recopilan de forma detallada las métricas y trazas obtenidas tras auditar la primera y segunda versión del sistema, respectivamente.
* **`rag_dataset_v3_qwen2.5_32b.json`**: conjunto de datos definitivo en formato JSON generado por la tercera iteración del pipeline utilizando el modelo instruccional `Qwen2.5-32B`. Este archivo actúa como el banco de pruebas consolidado bajo la taxonomía de cinco categorías descrita anteriormente.

### Scripts de generación de datasets
* **`generate_dataset_v1.py` y `generate_dataset_v2.py`**: primeras versiones del software de automatización destinadas a la extracción documental y formulación de pares de preguntas y respuestas (*QA*).
* **`generate_dataset_V3.py`**: versión óptima y más reciente del script de generación sintética, responsable de la creación del dataset de evaluación definitivo (V3) e inyección de los casos de estrés para las métricas de seguridad.

### Scripts de ejecución y evaluación
* **`evaluate_V1.py` y `evaluate_V2.py`**: herramientas encargadas de computar las métricas de rendimiento conversacional y fidelidad documental (*Context Precision*, *Faithfulness*, etc.) sobre los resultados de las dos primeras configuraciones del sistema.
* **`run_model_dataset.py`**: script de ejecución avanzado diseñado para cargar el dataset consolidado de la tercera versión (`rag_dataset_v3_qwen2.5_32b.json`). Su propósito es someter este banco de preguntas a inferencia utilizando configuraciones o parámetros alternativos del conducto RAG, permitiendo contrastar de forma empírica el impacto de los cambios arquitectónicos en la fase de síntesis final.