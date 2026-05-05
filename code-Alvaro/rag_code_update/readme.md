# RAG DIA

Sistema de consulta de guías académicas universitarias mediante RAG (Retrieval-Augmented Generation).

---

## Requisitos

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
| Servicio  | URL                          |
|-----------|------------------------------|
| Interfaz  | http://localhost:7860        |
| API docs  | http://localhost:9000/docs   |

---

## Carga de documentos

Una vez el stack está corriendo, editar `upload_documents.py` con la ruta a los documentos y ejecutar:

```bash
python upload_documents.py
```

Solo es necesario ejecutarlo una vez. Los datos persisten entre reinicios.

---

## Características

- **RAG-Fusion + Multi-Query**: genera 5 variantes de cada pregunta y fusiona los resultados con Reciprocal Rank Fusion para mejorar la recuperación.
- **Historial de conversación**: mantiene los últimos 10 turnos para resolver referencias contextuales.
- **Metadata jerárquica**: los documentos se indexan por curso, categoría y titulación, permitiendo filtros precisos.
- **Selector en cascada**: interfaz con selección Curso → Titulación → Documento.
- **Inspector de chunks**: visualiza los fragmentos exactos usados en la última respuesta.
