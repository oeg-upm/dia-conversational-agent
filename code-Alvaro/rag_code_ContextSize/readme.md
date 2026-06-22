# RAG Code Context Size

Este directorio contiene el RAG utilizado en el **Experimento 2: Evaluación del tamaño de contexto en el recuperador**.

Este sistema se emplea para realizar la ingesta de documentos y ejecutar las consultas necesarias para evaluar cómo afecta el número de chunks recuperados (`k`) a la calidad de las respuestas generadas por el sistema RAG.

Su función dentro del experimento es servir como RAG principal sobre el que se prueban distintos valores de `k`, manteniendo constante el resto de la configuración del sistema.

El experimento asociado se encuentra en:

```text
retriever_experiment/
```

---

## Requisitos

- Python >= 3.10
- Docker instalado y corriendo
- Servidor **Ollama** accesible con los modelos:
  - `qwen2.5:32b` — LLM
  - `qwen3-embedding:8b` — embeddings
- Configurar la URL del servidor en `backend/rag.py`

---

## Ejecución con Docker

### Primera vez

```bash
docker compose up --build
```

### Siguientes veces

```bash
docker compose up
```

### Parar

```bash
docker compose down
```

### Liberar espacio tras un rebuild

```bash
docker compose up --build && docker image prune -f
```

### Accesos

| Servicio | URL |
|---|---|
| Interfaz | http://localhost:7860 |
| API docs | http://localhost:9000/docs |

---

## Carga de documentos

Una vez el stack está corriendo, editar `upload_documents.py` con la ruta a los documentos y ejecutar:

```bash
python upload_documents.py
```

Este paso debe realizarse antes de generar el dataset base del experimento con:

```bash
python generate_base_dataset.py
```

Solo es necesario ejecutar la ingesta una vez. Los datos persisten entre reinicios.

---

## Características

- **RAG-Fusion + Multi-Query**: genera variantes de cada pregunta y fusiona los resultados con Reciprocal Rank Fusion para mejorar la recuperación.
- **Metadata jerárquica**: los documentos se indexan por curso, categoría y titulación, permitiendo filtros precisos.
- **Selector en cascada**: interfaz con selección Curso → Titulación → Documento.
- **Inspector de chunks**: visualiza los fragmentos exactos usados en la última respuesta.
