
# Generación y evaluación del RAG

## Fase 1: Generación del dataset (inspirado en RAGEval)

### 1. Extracción del esquema
Para cualquier guía docente que se introduzca en el sistema, el primer paso es pedirle a un LLM que extraiga la información clave y la convierta en un formato estructurado (como un JSON). Esto destila los hechos puros y evita que el modelo de generación se confunda con texto irrelevante o formatos PDF complejos.
De cada guía se deben extraer de forma determinista los siguientes campos estructurados:
- Descripción general: resumen del propósito de la asignatura y su objetivo principal.
- Datos generales: nombre de la asignatura, créditos (ECTS), semestre, carácter (optativa/obligatoria), curso académico e idioma.
- Profesorado: nombres, roles (coordinador/profesor), correos electrónicos, despachos y horarios de tutorías.
- Requisitos: conocimientos previos, asignaturas recomendadas o prerrequisitos obligatorios.
- Competencias y resultados: competencias generales/específicas a adquirir y los resultados de aprendizaje (RA) esperados.
- Temario: desglose estructurado de los bloques, módulos o temas de la asignatura.
- Cronograma: planificación temporal (semana a semana) detallando actividades de teoría, laboratorio, entregas y exámenes.
- Evaluación: criterios detallados para la evaluación continua y la prueba final (pesos, porcentajes, notas mínimas y tipos de pruebas).
- Recursos didácticos: bibliografía básica y de consulta, software recomendado u otros materiales de apoyo.


### 2. Generación de preguntas y respuestas (ground truth) según la taxonomía
Utilizando el esquema anterior, se instruye a un LLM para que genere pares de preguntas y respuestas ideales (question y ground_truth) basadas en la taxonomía definida en Know Your RAG.

- Preguntas factuales:
    - Definición: preguntas dirigidas a detalles específicos dentro de una referencia para probar la precisión de recuperación.
    - Ejemplo generado: "¿Quién es el coordinador de esta asignatura y en qué despacho son sus tutorías?"

- Preguntas de resumen (summarization):
    - Definición: preguntas que requieren respuestas exhaustivas que cubran toda la información relevante.
    - Ejemplo generado: "¿Podrías explicarme detalladamente cómo funciona el sistema de evaluación continua de esta materia?"

- Preguntas de razonamiento (multi-hop reasoning)
    - Definición: preguntas que implican relaciones lógicas entre eventos y detalles dentro de un documento.
    - Ejemplo generado: "Si no he cursado las asignaturas previas recomendadas en la guía, ¿es aconsejable que me matricule de todos modos?"

- Preguntas sin respuesta (unanswerable)
    - Definición: preguntas en las que no existe un fragmento de información correspondiente o la información es insuficiente para dar una respuesta.
    - Ejemplo generado: "¿Qué porcentaje de alumnos aprobó esta asignatura el año pasado?" o "¿Qué día exacto del mes es el examen final?" (si la guía solo menciona el mes).

La idea sería generar un 40% de preguntas factuales, un 20% de resumen, un 30% de razonamiento y un 10% de sin respuesta.

## Fase 2: Ejecución del sistema RAG
### 3. Inferencia y recopilación de contexto
Una vez construido el dataset con las preguntas generadas en el paso anterior, se introducen estas questions en el sistema RAG real. Por cada pregunta, el sistema debe devolver y registrar dos elementos fundamentales para la evaluación posterior:

- answer: la respuesta final que ha generado el RAG.
- contexts: los fragmentos de texto exactos (chunks) que el RAG ha recuperado de su base de datos vectorial para intentar responder a la pregunta.


## Fase 3: evaluación automatizada con RAGAS

### 4. Aplicación de métricas de evaluación
Con la tabla de datos completa (conteniendo question, ground_truth, answer y contexts), se utiliza el framework de código abierto Ragas para evaluar automáticamente el rendimiento del RAG. Se evalúan dos dimensiones:
- Métricas de generación (calidad de la respuesta):

    - Faithfulness: mide si la respuesta se basa estrictamente en el contexto recuperado, penalizando las alucinaciones.

    - Answer relevance: evalúa si la respuesta va al grano y contesta directamente a la pregunta original del alumno, sin aportar información innecesaria.

    - Answer correctness: compara semánticamente la respuesta generada por el RAG frente a la respuesta ideal que se generó en la Fase 1.

- Métricas de recuperación:

    - Context precision: evalúa la calidad del motor de búsqueda verificando si los fragmentos útiles (los que de verdad contienen la respuesta) aparecen en los primeros puestos de los resultados.

    - Context recall: mide si el motor de búsqueda ha sido capaz de encontrar toda la información necesaria en la base de datos para responder a la pregunta de forma completa.