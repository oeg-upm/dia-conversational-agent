# RAG Code Base

Este directorio contiene el RAG utilizado en el **Experimento 1: Verbalización de tablas en un sistema RAG** para trabajar con los **documentos originales**.

Su función es servir como sistema base de comparación frente a `rag_code_verb`, que utiliza los documentos verbalizados.

```text
rag_code_base → PDFs originales
rag_code_verb → PDFs verbalizados
```

## Configuración del experimento

```text
RAG: rag_code_base
Documentos utilizados: PDFs originales
Puerto de ChromaDB: 8002
Collection name: basic_rag_2
Backend/API: http://localhost:9000
Frontend: http://127.0.0.1:7860/
```
## Requisitos

Antes de ejecutar el RAG, es necesario tener instalados los siguientes componentes:

```text
Python >= 3.10
Docker instalado y corriendo
pip
```

Además, se recomienda utilizar un entorno virtual de Python para aislar las dependencias del proyecto.

El sistema necesita tener ChromaDB en ejecución mediante Docker antes de arrancar el backend, ya que la base vectorial se utiliza para almacenar y recuperar los documentos indexados.

También es necesario disponer de acceso al servidor donde se encuentran desplegados el modelo de lenguaje y el modelo de embeddings utilizados por el RAG. En caso de ejecutar el sistema en otro entorno, se deben revisar y modificar las URLs correspondientes dentro del código.

## Ejecución

### 1. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Levantar ChromaDB

Desde la carpeta donde esté el `docker-compose.yml`:

```bash
docker compose up -d
```

Para detenerlo:

```bash
docker compose down
```

### 3. Lanzar el backend

Desde la carpeta raíz del proyecto:

```bash
uvicorn backend.main:app --reload --port 9000
```

La API estará disponible en:

```text
http://localhost:9000/docs
```

### 4. Lanzar el frontend

```bash
cd frontend
python3 app.py
```

La interfaz estará disponible en:

```text
http://127.0.0.1:7860/
```

## Uso en el Experimento 1

Una vez levantado el sistema, se debe realizar la ingesta de los PDFs originales mediante el script:

```text
experiment_verbalize/upload_docs.py
```

Después, la evaluación se lanza desde:

```text
experiment_verbalize/
```

con:

```bash
python run_evaluation.py
```