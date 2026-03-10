# RAG (FastAPI + Docling + ChromaDB)

Sistema RAG local. El sistema lee, procesa y fragmenta documentos de forma inteligente para ofrecer respuestas precisas basadas estrictamente en el contexto proporcionado.

## Características

* **Parsing:** usa `Docling` (Hybrid Chunker) para extraer texto entendiendo la estructura visual del documento (tablas, títulos, párrafos) sin romper el contexto.
* **Embeddings:** utiliza el modelo `BAAI/bge-m3` para búsquedas semánticas avanzadas en español e inglés.
* **Búsqueda (RAG-Fusion):** implementa *Multi-Query* y *Reciprocal Rank Fusion (RRF)* para recuperar los fragmentos más relevantes.
* **Local:** el LLM funciona de manera local a través de LM Studio.
* **Interfaz web (gradio):** incluye un chat y un *Inspector* para auditar qué fragmentos exactos del documento usó el modelo para responder.

---


## Contenido
* **`backend.py`**: lógica de FastAPI y motor de búsqueda RAG-Fusion.
* **`frontend.py`**: interfaz de usuario interactiva creada con Gradio.
* **`Dockerfile.backend`**: imagen del backend con soporte para aceleración GPU (para el embedding).
* **`Dockerfile.macos.backend`**: imagen del backend para macOS.
* **`Dockerfile.frontend`**: imagen ligera para el despliegue de la interfaz web.
* **`docker-compose.yml`**: orquestador de servicios, redes y volúmenes de datos.
* **`docker-compose.macos.yml`**: lo mismo pero para macOS.
* **`requirements.txt`**: listado de dependencias y librerías necesarias del sistema.

---

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado:
1. **Python 3.10+**
2. **Docker** (para levantar los servicios).
3. **LM Studio** corriendo un modelo local (ej. `llama-3.2-3b-instruct`) con el servidor local activado en `http://127.0.0.1:1234/v1`.


---

## Instalación y uso

### 1.1 Levantar el sistema (Windows)
Ejecuta el siguiente comando en el directorio ```/rag-code``` para construir las imágenes e iniciar los servicios en segundo plano:

```bash
docker-compose up --build -d
```

### 1.2 Levantar el sistema (macOS, no lo he probado porque tengo windows, debería ir)
Para macOS, se eliminan las dependencias de NVIDIA/CUDA y se usa una imagen de Python pura, por lo que hay un archivo `dockerfile` diferente del backend y un `docker-compose.macos.yml`. Para levantarlo, ejecuta el siguiente comando en el directorio `/rag-code` para construir las imágenes e iniciar los servicios en segundo plano:

```bash
docker-compose -f docker-compose.macos.yml up --build -d
```

### 2. Acceso a las interfaces
Una vez levantado, puedes acceder a:

Frontend (chat): http://localhost:7860

Backend (API Docs): http://localhost:8001/docs

ChromaDB: http://localhost:8000/docs


### 3. Mantenimiento y logs
Al estar en segundo plano, puedes monitorear qué está haciendo el sistema:

- Ver logs del Backend:
```bash
docker logs -f rag_backend
```

- Ver logs del Frontend:
```bash
docker logs -f rag_frontend
```

- Detener el sistema:
```bash
docker-compose down
```

- Borrar la base de datos para empezar de cero:
```bash
docker-compose down -v
```