# OPTIMIZACIÓN DE PARÁMETROS DEL RAG

## RETRIEVAL

### ¿Qué estrategia de chunking es más apropiada para el almacenamiento y retrieval de información?
Comparar resultados y scores (calidad del retrieval) de diferentes estrategias de chunking (semántico vs fixed size) y diferentes tamaños.

Referencias:  

### ¿Qué hace mejor a un modelo de embeddings? ¿Un modelo más grande capta mejor la semántica?
Comparar si hay modelos de embeddings más adecuados para un tema, sector, lenguaje... Lenguaje informático. 
Se representa mejor la información en un embedding de más dimensiones, y se hace un mejor retrieval?  

### ¿Añadir un modelo de reranking mejora la calidad de los resultados de retrieval?
Comparar resultados y scores (calidad del retrieval) con y sin modelo de reranking.

## GENERACIÓN

### ¿Se pierde calidad en la generación de un LLM cuantizado respecto a su versión sin cuantizar?

### ¿Es mejor la generación de un modelo más grande (para este caso de uso)?
Cuanto se gana en calidad y vigilar también lo que se pierde en latencia (trade-off)

## ¿Cuál es el tamaño ideal de contexto que pasarle a un LLM para responder? ¿Cuál es el número "ideal" de chunks a devolver?

## APLICACIÓN

### ¿Mejora la calidad de la respuesta incorporar estrategias como reformular la pregunta?

### ¿Mejora la calidad de la respuesta incorporar estrategias como multi-query? 
Vigilar el coste de latencia que supone. 

### ¿Cuál es la mejor forma de gestionar preguntas fuera de contexto en un chatbot? 
Identificar primero si la pregunta require RAG o no (con un agente). 

### ¿Cómo se puede incorporar feedback humano en la aplicación de forma semi-automática?
Crear unos tests de evaluación. 
