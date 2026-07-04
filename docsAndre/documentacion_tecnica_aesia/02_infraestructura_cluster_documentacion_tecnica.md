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

| Modelo | Tipo | Parámetros | Función en el sistema |
|--------|------|-----------|----------------------|
| `qwen2.5:32b` | LLM generativo | 32B | Generación de respuestas RAG, generación de consultas múltiples (Multi-Query), juez LLM en evaluación RAGAS |
| `qwen3-embedding:8b` | Modelo de embeddings | 8B | Vectorización de chunks y consultas para búsqueda semántica |

*Nota: Durante los experimentos de evaluación se utilizaron modelos adicionales con diferentes configuraciones de embeddings. Ver [04_evaluadores_documentacion_tecnica.md](04_evaluadores_documentacion_tecnica.md) para el detalle completo.*

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
