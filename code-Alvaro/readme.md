# Proyecto

- Se ha incorporado una arquitectura modular, separando el frontend y el backend mediante una API desarrollada con FastAPI.
- Se ha implementado ChromaDB ejecutándose en un contenedor Docker.
- Se ha añadido el endpoint `list_documents`, que permite visualizar los documentos procesados y almacenados en ChromaDB.



---

# Ejecución del Proyecto

## 1. Iniciar el backend

Desde la carpeta raíz del proyecto, ejecutar:

```bash
uvicorn backend.main:app --reload --port 9000
```

Para acceder a la Api:

```
http://localhost:9000/docs
```

---

## 2. Iniciar el frontend

Acceder a la carpeta `frontend` y ejecutar:

```bash
python3 app.py
```
acceder a la interfaz

```
 http://127.0.0.1:7860/
 ```
---


## 3. Requisitos

- LM Studio, el LLM esta corriendo en local (llama-3.2-3b-instruct) con el servidor en http://127.0.0.1:1234
- Correr chromadb en docker

## 4. Recomendado utilizar entorno virtual

Crear entorno virtual en python

```bash
python3 -m venv venv
```
Acceder al entorno virtual

```bash
source venv/bin/activate
```
Instalar todas las dependencias del proyecto

```bash
pip install -r requirements.txt
```
