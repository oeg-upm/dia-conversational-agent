# Experimento 2: Evaluación del tamaño óptimo de contexto en un sistema RAG

Este experimento analiza el impacto del número de fragmentos recuperados por el sistema RAG sobre la calidad de las respuestas generadas.

El objetivo principal es estudiar cómo afecta el parámetro `k`, es decir, el número de chunks recuperados, al rendimiento del sistema. Para ello, se evalúa el mismo sistema RAG con distintos valores de `k` y se comparan las métricas obtenidas.

Además, el experimento permite realizar esta evaluación bajo dos configuraciones de recuperación diferentes:

```text
Retrieve puro
Multi-Query
```

Este experimento se ejecuta sobre el RAG:

```text
rag_code_ContextSize
```

El código específico del experimento se encuentra en:

```text
experiment_ContextSize/
```

---

## 1. Descripción general del experimento

El experimento se divide en cuatro pasos principales:

```text
1. Levantar el sistema RAG actualizado.
2. Realizar la ingesta de documentos mediante upload_documents.py.
3. Generar el dataset base mediante generate_base_dataset.py.
4. Ejecutar el experimento completo mediante experiment_h6.py.
```

El flujo general consiste en levantar el RAG, cargar los documentos en la base vectorial, generar un dataset base de evaluación y ejecutar posteriormente el experimento completo para distintos valores de `k`.

---

## 2. RAG utilizado

Este experimento utiliza únicamente el siguiente RAG:

```text
rag_code_ContextSize
```

Este RAG incorpora una estrategia de recuperación más avanzada basada en:

```text
RAG-Fusion
Multi-Query
Reciprocal Rank Fusion
Metadata jerárquica
Filtrado por curso, titulación y documento
```

Para reproducir el experimento, primero se debe levantar este RAG siguiendo las instrucciones de su propio README.

---

## 3. Ejecución del RAG

Desde la carpeta del RAG actualizado:

```bash
cd code-Alvaro/rag_code_ContextSize
```

La primera vez, se recomienda ejecutar:

```bash
docker compose up --build
```

En ejecuciones posteriores:

```bash
docker compose up
```

Para detener el sistema:

```bash
docker compose down
```

Una vez levantado, el sistema debe exponer los siguientes servicios:

```text
Frontend: http://localhost:7860
Backend/API: http://localhost:9000
Documentación API: http://localhost:9000/docs
```

El endpoint principal utilizado por el experimento es:

```text
POST /chat
```

El endpoint utilizado para la ingesta de documentos es:

```text
POST /upload
```

---

## 4. Configuración del modelo de lenguaje

El RAG utiliza un modelo de lenguaje y un modelo de embeddings servidos mediante un servidor externo, en este caso asociado al entorno de la universidad.

Por ello, antes de ejecutar el experimento, es necesario revisar las URLs configuradas en el código.

Los modelos utilizados son:

```text
LLM: qwen2.5:32b
Embeddings: qwen3-embedding:8b
```

En el RAG, la URL del servidor debe revisarse en:

```text
rag_code_ContextSize/backend/rag.py
```

En los scripts del experimento también deben revisarse las URLs del modelo en:

```text
experiment_ContextSize/generate_base_dataset.py
experiment_ContextSize/evaluate.py
```

Ejemplo de configuración del LLM:

```python
LLM_CONFIG = {
    "model": "qwen2.5:32b",
    "base_url": "http://URL_DEL_SERVIDOR/v1",
    "api_key": "not_required",
    "temperature": 0.7
}
```

Ejemplo de configuración para evaluación con RAGAS:

```python
ollama_url = "http://URL_DEL_SERVIDOR_OLLAMA:11434"
```

Si se ejecuta el experimento en otro entorno, estas URLs deben modificarse para apuntar al servidor correspondiente.

---

## 5. Ingesta de documentos

Una vez levantado `rag_code_ContextSize`, se deben cargar los documentos en ChromaDB mediante el script:

```text
upload_documents.py
```

Este script se encuentra en:

```text
rag_code_ContextSize/upload_documents.py
```

El script recorre una carpeta raíz de guías docentes y sube automáticamente todos los archivos PDF encontrados al backend del RAG.

La ruta principal debe configurarse en la variable:

```python
BASE_DIR = r"/Guías aprendizaje"
```

El endpoint de subida se configura mediante:

```python
BACKEND_URL = "http://localhost:9000/upload"
```

Para ejecutar la ingesta:

```bash
cd code-Alvaro/rag_code_ContextSize
python upload_documents.py
```

Antes de ejecutar el resto del experimento, es importante comprobar que los documentos se han cargado correctamente en ChromaDB.

---

## 6. Generación del dataset base

Una vez realizada la ingesta de documentos, se debe generar el dataset base mediante:

```text
generate_base_dataset.py
```

Este script se encuentra en:

```text
experiment_ContextSize/generate_base_dataset.py
```

Para ejecutarlo:

```bash
cd code-Alvaro/experiment_ContextSize
python generate_base_dataset.py
```

El script genera el dataset base de evaluación a partir de los documentos indexados en ChromaDB.

La configuración principal del script es:

```python
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_dia"

OUTPUT_FILE = "datasets/base_dataset.json"
N_SAMPLES = 50
```

El archivo generado es:

```text
datasets/base_dataset.json
```

Este dataset será utilizado posteriormente por `experiment_h6.py` para consultar el RAG con distintos valores de `k`.

---

## 7. Ejecución del experimento completo

Una vez generado el dataset base, se ejecuta el experimento completo mediante:

```text
experiment_h6.py
```

Para lanzarlo:

```bash
cd code-Alvaro/experiment_ContextSize
python experiment_h6.py
```

Este script carga:

```text
datasets/base_dataset.json
```

y realiza consultas al backend del RAG para distintos valores de `k`.

La configuración principal es:

```python
BACKEND_URL = "http://localhost:9000"
BASE_DATASET_PATH = "datasets/base_dataset.json"
K_VALUES = [1, 3, 5, 7, 10, 15, 20]
```

Por cada valor de `k`, el script genera un dataset con las respuestas y contextos obtenidos del RAG y ejecuta la evaluación correspondiente.

Además de modificar los valores de `k`, este experimento permite evaluar el tamaño de contexto bajo dos configuraciones distintas de recuperación:

```text
Retrieve puro
Multi-Query
```

Esta configuración se modifica en `experiment_h6.py`, concretamente en la petición realizada al endpoint `/chat`.

Para ejecutar el experimento con **retrieve puro**, se debe establecer:

```python
use_multiquery = False
```

Para ejecutar el experimento con **Multi-Query**, se debe establecer:

```python
use_multiquery = True
```

Por tanto, el experimento puede repetirse dos veces, manteniendo los mismos valores de `k`, pero cambiando la estrategia de recuperación utilizada. Esto permite comparar si el tamaño óptimo de contexto varía entre una recuperación directa y una recuperación basada en Multi-Query.

---

## 8. Valores de k evaluados

El experimento evalúa los siguientes valores de `k`:

```text
k = 1
k = 3
k = 5
k = 7
k = 10
k = 15
k = 20
```

Estos valores permiten estudiar cómo afecta el tamaño del contexto recuperado a la calidad final del sistema.

Un valor bajo de `k` puede limitar la cantidad de información recuperada, reduciendo el recall. Sin embargo, un valor demasiado alto puede introducir ruido, aumentar la latencia y favorecer problemas como el efecto `lost-in-the-middle`.

---

## 9. Datasets generados durante el experimento

Por cada valor de `k`, el script genera un dataset con las respuestas y contextos obtenidos del RAG.

Ejemplo de salidas:

```text
datasets/dataset_k5_multi.json
datasets/dataset_k7_multi.json
datasets/dataset_k10_multi.json
datasets/dataset_k15_multi.json
datasets/dataset_k20_multi.json
```

Estos archivos contienen el dataset base enriquecido con las respuestas generadas por el RAG, los contextos recuperados y la latencia asociada.

---

## 10. Evaluación con RAGAS

La evaluación se realiza mediante RAGAS.

Las métricas utilizadas son:

```text
faithfulness
answer_relevancy
context_precision
context_recall
```

Estas métricas permiten analizar tanto la calidad de la respuesta generada como la calidad de los contextos recuperados por el sistema RAG.

---

## 11. Resultados generados

Los resultados de evaluación se guardan en la carpeta:

```text
results/
```

Ejemplo de archivos generados:

```text
results/evaluation_k5_multi_summ.csv
results/evaluation_k7_multi_summ.csv
results/evaluation_k10_multi_summ.csv
results/evaluation_k15_multi_summ.csv
results/evaluation_k20_multi_summ.csv
```

Cada CSV contiene las métricas obtenidas para cada valor de `k`.

La comparación final se realiza observando las medias de las métricas para cada configuración.

---

## 12. Resumen del flujo completo

El flujo completo para reproducir el experimento es:

```text
1. Entrar en rag_code_ContextSize.
2. Levantar el RAG con docker compose up o docker compose up --build.
3. Revisar la URL del modelo de lenguaje y del modelo de embeddings.
4. Configurar la ruta BASE_DIR en upload_documents.py.
5. Ejecutar upload_documents.py para realizar la ingesta de documentos.
6. Entrar en experiment_ContextSize.
7. Revisar CHROMA_HOST, CHROMA_PORT y COLLECTION_NAME en generate_base_dataset.py.
8. Ejecutar generate_base_dataset.py para crear datasets/base_dataset.json.
9. Revisar BACKEND_URL, K_VALUES y use_multiquery en experiment_h6.py.
10. Ejecutar experiment_h6.py.
11. Revisar los datasets generados en datasets/.
12. Revisar los resultados generados en results/.
13. Comparar las métricas medias para cada valor de k.
```

---

## 13. Comandos principales

Desde la raíz del repositorio:

```bash
cd code-Alvaro/rag_code_ContextSize
docker compose up --build
```

En otra terminal, realizar la ingesta:

```bash
cd code-Alvaro/rag_code_ContextSize
python upload_documents.py
```

Después, generar el dataset base:

```bash
cd ../experiment_ContextSize
python generate_base_dataset.py
```

Finalmente, ejecutar el experimento completo:

```bash
python experiment_h6.py
```

---

## 14. Consideraciones de reproducibilidad

Para que el experimento sea reproducible, se recomienda mantener constantes los siguientes elementos:

```text
Modelo generativo
Modelo de embeddings
Colección de ChromaDB
Documentos indexados
Número de muestras del dataset base
Valores de k evaluados
Métricas de evaluación
Servidor del modelo
Versión de las librerías
```

En concreto, la configuración base del experimento es:

```text
RAG utilizado: rag_code_ContextSize
Backend RAG: http://localhost:9000
Chroma host: localhost
Chroma port: 8000
Collection name: rag_dia
Dataset base: datasets/base_dataset.json
Número de muestras: 50
Valores de k: 1, 3, 5, 7, 10, 15, 20
Configuración de recuperación: retrieve puro o Multi-Query
LLM: qwen2.5:32b
Embeddings: qwen3-embedding:8b
```

---

## 15. Advertencias importantes

### 15.1. Rutas locales

El código original utiliza rutas absolutas del entorno local de desarrollo.

Antes de ejecutar el experimento en otro equipo, revisar especialmente:

```text
rag_code_ContextSize/upload_documents.py
experiment_ContextSize/generate_base_dataset.py
experiment_ContextSize/evaluate.py
```

En particular, se debe modificar:

```python
BASE_DIR = r"/home/alvaro/Escritorio/Guías aprendizaje"
```

para que apunte a la carpeta real donde estén almacenadas las guías docentes.

---

### 15.2. URL del servidor del modelo

El modelo de lenguaje y el modelo de embeddings están desplegados en un servidor externo de la universidad.

Si se ejecuta el experimento fuera de ese entorno, se deben modificar las URLs correspondientes.

Revisar:

```text
rag_code_ContextSize/backend/rag.py
experiment_ContextSize/generate_base_dataset.py
experiment_ContextSize/evaluate.py
```

Variables importantes:

```text
base_url
ollama_url
```

---

### 15.3. ChromaDB y colección

El script `generate_base_dataset.py` se conecta a ChromaDB usando:

```python
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_dia"
```

Estos valores deben coincidir con la configuración real del RAG levantado en `rag_code_ContextSize`.

Si el puerto o el nombre de la colección cambian, también deben actualizarse en el script.

---

### 15.4. Ingesta previa obligatoria

Antes de ejecutar `generate_base_dataset.py`, es obligatorio haber realizado la ingesta de documentos.

Si ChromaDB está vacío, el script no podrá generar el dataset base porque no tendrá documentos indexados.

El orden correcto es:

```text
1. Levantar RAG.
2. Ingerir documentos.
3. Generar dataset base.
4. Ejecutar experimento.
```

---

### 15.5. Limpieza de la base vectorial

Antes de repetir el experimento desde cero, se recomienda comprobar si ChromaDB ya contiene documentos de ejecuciones anteriores.

Si se mezclan documentos antiguos con documentos nuevos, los resultados pueden verse alterados.

Para una comparación limpia, se recomienda eliminar la colección anterior o reconstruir el volumen de ChromaDB antes de una nueva ingesta.

---

### 15.6. Tiempo de ejecución

El experimento puede tardar bastante tiempo, ya que para cada valor de `k` se consulta el RAG con todas las preguntas del dataset y posteriormente se ejecuta la evaluación con RAGAS.

Además, la evaluación puede configurarse con un único worker para evitar sobrecargar la VRAM del servidor:

```python
max_workers = 1
```

Esto reduce el riesgo de errores por memoria, aunque aumenta el tiempo total de ejecución.

---

## 16. Finalidad del experimento

Este experimento permite estudiar el equilibrio entre la cantidad de contexto recuperado y la calidad final de la respuesta generada por el sistema RAG.

La hipótesis principal es que aumentar el valor de `k` puede mejorar el `context_recall`, ya que el sistema recupera más información potencialmente relevante. Sin embargo, a partir de cierto punto, añadir más chunks puede reducir el `context_precision`, aumentar la latencia y perjudicar la calidad de la respuesta al introducir información redundante o poco relevante.

Por tanto, el análisis se centra en identificar el valor de `k` que ofrece el mejor equilibrio entre:

```text
Calidad de recuperación
Calidad de generación
Precisión del contexto
Recall del contexto
Latencia
