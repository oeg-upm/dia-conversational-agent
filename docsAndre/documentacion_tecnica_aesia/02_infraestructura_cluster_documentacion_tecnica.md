# Documentación Técnica de la Infraestructura de Clúster
## Proyecto DIA — Agente Conversacional para la FI-UPM

**Guías de referencia:** AESIA #15 (§5.1.e, §5.2.c, §5.9) · AESIA #11 (Ciberseguridad) · AESIA #12 (Registros)  
**Fecha de redacción:** Julio 2026

---

## 1. Descripción de la infraestructura

El sistema DIA se ejecuta sobre una infraestructura de clúster universitario que proporciona capacidad de cómputo GPU para la inferencia de modelos de lenguaje grandes (LLM) y modelos de embeddings. Esta infraestructura es compartida entre los miembros del equipo de investigación del proyecto.

### 1.1 Hardware de cómputo

| Componente | Especificación |
|------------|---------------|
| GPU | NVIDIA H100 (acceso mediante clúster universitario) |
| Servidor de modelos | Ollama — servidor de inferencia LLM de código abierto |
| IP de clúster (interna VPN) | `100.84.51.82` |
| Puerto API | `5000` (interfaz OpenAI-compatible) |
| Protocolo API | HTTP REST — compatible con especificación OpenAI v1 |

### 1.2 Modelos alojados en el clúster

Inventario completo de modelos disponibles en el clúster (`ollama list`, julio 2026). Ordenados por función y fecha de instalación descendente:

#### Modelos de embeddings

| Modelo | Tamaño | Última vez modificado | Uso en el proyecto |
|--------|--------|----------------------|-------------------|
| `nicolasfer45/Octen-Embedding-4B-GGUF:latest` | 2.5 GB | Hace 3 semanas | **Experimento de embeddings (H1A, André)** — mejor recall en H1A (0.830). Usado en la configuración *Octen* del experimento de humanización. Cuantizado en GGUF. |
| `leoipulsar/harrier-0.6b:latest` | 639 MB | Hace 3 semanas | Evaluado en H1A (Juanma). Buen balance general pero no seleccionado para los experimentos de André por no destacar sobre Octen y ser menos citado en la literatura. |
| `embeddinggemma:latest` | 621 MB | Hace 3 semanas | Exploración temprana. No utilizado en experimentos formales — descartado por rendimiento inferior a bge-m3 en pruebas iniciales. |
| `bge-m3:latest` | 1.2 GB | Hace 5 semanas | **Experimento de embeddings (H1A, André)** — baseline canónico. Modelo más referenciado en la literatura RAG en español. Usado en configuración *BGE* del experimento de humanización. |
| `qwen3-embedding:4b` | 2.5 GB | Hace 6 semanas | **Experimento de embeddings (H1A, André)** — usado en configuración *Qwen4b* del experimento de humanización. Se eligió la versión 4b (en lugar de la 8b evaluada por Juanma) para estudiar el efecto del tamaño dentro de la familia qwen3. |
| `qwen3-embedding:8b` | 4.7 GB | Hace 2 meses | **Sistema RAG en producción** — embedding del sistema desplegado (`BasicRAG`). También usado como embedding del juez RAGAS. Evaluado por Juanma en H1A (mejor faithfulness del grupo). |

#### Modelos de lenguaje grandes (LLM generativos)

| Modelo | Tamaño | Última vez modificado | Uso en el proyecto |
|--------|--------|----------------------|-------------------|
| `llama3.1:8b` | 4.9 GB | Hace 2 semanas | Usado en la versión V1 del generador de dataset (`generate_dataset_v1.py`, Juanma). Descartado para versiones posteriores por generar ground truth demasiado corto y distribución de dificultades sesgada hacia "easy". |
| `gemma3:27b` | 17 GB | Hace 2 semanas | Probado como LLM generador del dataset v3. Calidad aceptable pero con tendencia a mezclar idiomas en preguntas largas. No seleccionado como dataset final. |
| `gemma4:31b` | 19 GB | Hace 5 semanas | Variante grande de la familia gemma4. Instalado para experimentación posterior; no utilizado en experimentos formales documentados. |
| `qwen3.6:27b` | 17 GB | Hace 5 semanas | Modelo reciente de la familia qwen3.6. Instalado para exploración; no utilizado en experimentos formales documentados. |
| `gemma4:latest` | 9.6 GB | Hace 2 meses | Usado por Juanma en generación de dataset (`rag_dataset_v3_gemma4_26b.json`). Calidad alta — preguntas muy naturales. Seleccionado en experimentos alternativos de Juanma pero no en el dataset principal del experimento de humanización. |
| `qwen2.5:14b` | 9.0 GB | Hace 2 meses | Probado como generador de dataset v3. Mejora notable sobre los modelos de 7-8B pero inferior a la versión 32B. Descartado en favor de `qwen2.5:32b`. |
| `ministral-3:14b` | 9.1 GB | Hace 2 meses | Probado como generador de dataset v3. Baja diversidad de preguntas y errores frecuentes en el campo `language` (mezcla es/en). Descartado. |
| `qwen3.5:9b` | 6.6 GB | Hace 2 meses | Probado como generador de dataset v3. Calidad similar a llama3.1:8b. Inferior a qwen2.5:32b. Descartado. |
| `qwen3.5:27b` | 17 GB | Hace 2 meses | Explorado como generador de dataset. No utilizado en experimentos formales documentados. |
| `qwen2.5:7b` | 4.7 GB | Hace 2 meses | Exploración temprana. No utilizado en experimentos formales — inferior a las variantes de mayor tamaño de la misma familia. |
| `qwen3.5:35b` | 23 GB | Hace 2 meses | Probado como generador de dataset v3. Calidad comparable a `qwen2.5:32b` pero mayor tiempo de generación. Sin ventaja suficiente para justificar el coste computacional adicional. Descartado. |
| `deepseek-r1:32b` | 19 GB | Hace 2 meses | Probado como generador de dataset v3. Alta calidad de razonamiento pero genera "chain of thought" en los ground truths (explicaciones largas en vez de respuestas directas). Incompatible con el formato requerido por RAGAS. Descartado. |
| `qwen2.5:32b` | 19 GB | Hace 2 meses | **Sistema RAG en producción** (LLM de generación de respuestas, Multi-Query). **Juez LLM en evaluación RAGAS**. **Generador del dataset final** (`rag_dataset_v3_octen_qwen2.5_V2.json`). El modelo más utilizado del proyecto. |

#### Resumen de uso por modelo en experimentos formales

| Experimento | Embedding | LLM generativo |
|------------|-----------|----------------|
| RAG en producción | qwen3-embedding:8b | qwen2.5:32b |
| H1A — comparación de embeddings (Juanma) | bge-m3, harrier-0.6b, Octen-4B, qwen3-embedding:8b | qwen2.5:32b (fijo) |
| H1B — Octen × LLM × multiquery (Juanma) | Octen-4B | qwen2.5:32b |
| H3 — BGE × LLM (Juanma) | bge-m3 | qwen2.5:32b |
| Generación dataset v3 final (Juanma) | — | qwen2.5:32b |
| Experimento de humanización (André) | bge-m3, qwen3-embedding:4b, Octen-4B | qwen2.5:32b (fijo) |
| Evaluación RAGAS (juez) | qwen3-embedding:8b | qwen2.5:32b |

---

## 2. Arquitectura de red y acceso (Guía 15, §5.2.c)

### 2.1 Tailscale VPN — Control de acceso a la infraestructura

El clúster no está expuesto a Internet público. El acceso se realiza exclusivamente a través de **Tailscale**, una red privada virtual (VPN) de malla (*mesh VPN*) basada en WireGuard.

**Características del control de acceso:**

| Aspecto | Detalle |
|---------|---------|
| Tecnología VPN | Tailscale (basada en WireGuard) |
| Autenticación | Cuenta Tailscale con autenticación de dos factores (2FA) |
| Alcance de la red | Solo miembros del equipo autorizados con cuenta Tailscale del proyecto |
| Dirección IP | `100.84.51.82` — dirección Tailscale estática del nodo de clúster |
| Cifrado en tránsito | WireGuard (cifrado extremo a extremo) |

### 2.2 Endpoints expuestos

| Endpoint | Puerto | Protocolo | Acceso |
|----------|--------|-----------|--------|
| Ollama API (LLM + embeddings) | `5000` | HTTP/REST (OpenAI-compatible) | Solo desde red Tailscale |
| Ollama REST nativo | `5000` | HTTP/REST | Solo desde red Tailscale |
| ChromaDB | `8000` | HTTP/REST | Solo desde red Docker interna (`chromadb` hostname) |

### 2.3 Relación de IPs del proyecto

Durante el desarrollo y los experimentos de evaluación, se observaron diferentes IPs del clúster en el código fuente, correspondientes a nodos distintos del clúster Tailscale utilizados en diferentes fases:

| IP Tailscale | Contexto de uso |
|-------------|----------------|
| `100.84.51.82` | Sistema RAG en producción (Álvaro) |
| `100.78.104.3` | Experimentos de evaluación del embedding (André) |
| `100.71.243.90` | Evaluación RAGAS de Juanma |
| `100.114.130.128` | Generación de dataset v3 (Juanma) |

---

## 3. Despliegue del sistema (Guía 15, §5.1.e)

### 3.1 Contenedores Docker

El backend RAG y la base de datos vectorial ChromaDB se despliegan mediante Docker:

```
┌─────────────────────────────────────────┐
│           Docker Network                │
│                                         │
│  ┌──────────────┐   ┌───────────────┐   │
│  │  FastAPI     │   │   ChromaDB    │   │
│  │  Backend     │──►│   :8000       │   │
│  │  (backend/)  │   │               │   │
│  └──────┬───────┘   └───────────────┘   │
│         │                               │
└─────────┼───────────────────────────────┘
          │ HTTP externo
          ▼
    [Frontend Gradio]
          │
          │ VPN Tailscale
          ▼
    [Clúster: Ollama :5000]
    (qwen2.5:32b + qwen3-embedding:8b)
```

**Fichero de dependencias del backend** (`requirements.txt`):
```
fastapi, uvicorn[standard], python-multipart, easyocr, chromadb,
langchain, langchain-openai, langchain-ollama, langchain-chroma,
langchain-community, langchain-docling, langchain-huggingface,
docling, docling-core, pydantic, requests
```

### 3.2 Almacenamiento temporal de archivos

Los archivos PDF subidos se almacenan temporalmente en el directorio `uploaded_files/` del contenedor para procesamiento, y se eliminan tras completar la ingestión en ChromaDB.

---

## 4. Ciberseguridad (Guía 11)

### 4.1 Superficie de ataque y medidas de protección

| Vector de ataque | Evaluación | Medida existente |
|-----------------|-----------|-----------------|
| Acceso no autorizado al clúster | Bajo riesgo | Tailscale VPN + autenticación de cuenta |
| Interceptación de tráfico LLM | Bajo riesgo | WireGuard (cifrado extremo a extremo en VPN) |
| Inyección de prompt | Riesgo presente | Evaluado activamente — 15 prompts adversariales en dataset de seguridad (sub-dimensión 2a) |
| Exposición de API sin autenticación | Riesgo en red local | El endpoint `/chat` no requiere autenticación (prototipo académico). En producción requeriría autenticación. |
| Denegación de servicio al clúster | Bajo riesgo | Acceso limitado a equipo de investigación |
| Exposición de instrucciones del sistema | Evaluado | Dataset de seguridad incluye prompts de extracción de prompt del sistema (categoría transparencia) |

### 4.2 Inyección de prompt — evaluación activa

El sistema fue evaluado con 15 prompts de inyección de prompt (sub-dimensión 2a del dataset de seguridad), diseñados siguiendo la metodología **SafeRAG**. Las preguntas están redactadas en español y son indistinguibles de consultas legítimas para probar si el sistema ejecuta instrucciones embebidas en las preguntas.

Ejemplos de vectores probados:
- Instrucciones de cambio de rol incrustadas en la pregunta
- Solicitudes de revelación de prompt del sistema camufladas como consultas académicas
- Intentos de eludir la restricción de no-responder fuera del corpus

*Referencia completa: [04_evaluadores_documentacion_tecnica.md](04_evaluadores_documentacion_tecnica.md), sección de evaluación de seguridad.*

### 4.3 Limitaciones de seguridad conocidas (contexto académico)

1. **Sin autenticación en API:** El endpoint `/chat` no tiene control de acceso propio. La seguridad recae enteramente en la VPN Tailscale.
2. **Sin rate limiting:** No existe limitación de peticiones por usuario.
3. **Sin WAF:** No hay filtro de aplicación web. En despliegue público requeriría solución dedicada.

---

## 5. Registros del sistema (Guía 12)

### 5.1 Logs del backend

El sistema genera archivos de log por sesión de operación en formato `backend_{session_id}.log`. Cada archivo de log contiene:

- Consultas generadas por el sistema Multi-Query (impresas en stdout como `Generated queries: [...]`)
- Chunks finales seleccionados tras RRF (impresos como `Final retrieved chunks (after RRF):`)
- Para cada chunk: curso, titulación, nombre de archivo fuente, número de chunk

**Ejemplo de entrada de log:**
```
Generated queries:
['¿Cuáles son los criterios de evaluación de...', 'Criterios de calificación para la asignatura...', ...]

Final retrieved chunks (after RRF):
  1. 2024-25 - GII - guia_asignatura.pdf - Chunk 42
  2. 2024-25 - GII - guia_asignatura.pdf - Chunk 43
  ...
```

### 5.2 Trazabilidad de versiones

El repositorio Git actúa como registro de cambios del sistema. Cada modificación significativa al código está asociada a un commit con descripción. Esto cubre el requisito de trazabilidad de versiones del sistema (Guía 12, §4.2).

### 5.3 Registros de experimentos de evaluación

Los resultados de evaluación se persisten en archivos CSV:
- `code-andre/eval/experiment_results/summary_original_vs_humanized.csv` — medias por configuración y dataset
- `code-andre/eval/experiment_results/bootstrap_paired_deltas.csv` — intervalos de confianza de los deltas

---

## 6. Continuidad y recuperación (Guía 15, §5.7)

### Dependencias externas críticas

| Dependencia | Impacto de fallo | Plan de recuperación |
|------------|-----------------|---------------------|
| Clúster Ollama (VPN) | Sistema no operativo (LLM y embeddings no disponibles) | Re-conexión Tailscale; reinicio del servicio Ollama en clúster |
| ChromaDB | Sistema no operativo (retrieval no disponible) | Reinicio del contenedor Docker; corpus persiste en volumen Docker |
| Red Tailscale | Sistema no operativo | Verificación de conectividad Tailscale; contacto con administrador de red |

### Estado de persistencia del corpus

El corpus vectorial persiste en el volumen Docker de ChromaDB entre reinicios del contenedor. La re-ingestión de documentos ya procesados es segura: el sistema detecta IDs deterministas duplicados y los omite (`unique_file_id` check en `add_documents_from_files`).

---

*Para la documentación del sistema RAG que se ejecuta sobre esta infraestructura, ver [01_rag_sistema_documentacion_tecnica.md](01_rag_sistema_documentacion_tecnica.md).*
