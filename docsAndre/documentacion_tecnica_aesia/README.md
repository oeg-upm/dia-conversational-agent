# Documentación Técnica AESIA — Proyecto DIA
## Agente Conversacional RAG para la Facultad de Informática (UPM)

**Marco de referencia:** Guías AESIA del Sandbox Regulatorio de IA de España (#05–#15)  
**Contexto:** Proyecto de investigación académica — prototipo, sin despliegue en producción pública  
**Fecha:** Julio 2026

---

## Índice de documentos

| Documento | Guías AESIA relacionadas | Contenido |
|-----------|--------------------------|-----------|
| [00 — Alcance y Limitaciones](00_alcance_y_limitaciones.md) | Todas | Qué aplica, qué no aplica y por qué. Análisis honesto del contexto académico. |
| [01 — Sistema RAG](01_rag_sistema_documentacion_tecnica.md) | #15, #06, #08, #05 | Backend RAG-Fusion completo: arquitectura, pipeline, endpoints, restricciones. |
| [02 — Infraestructura de Clúster](02_infraestructura_cluster_documentacion_tecnica.md) | #15, #11, #12 | Clúster GPU (H100), Ollama, Tailscale VPN, Docker, logging. |
| [03 — Generador de Dataset](03_generador_dataset_documentacion_tecnica.md) | #15, #07 | Generador sintético de QA pairs: corpus, esquema, pipeline, versiones. |
| [04 — Evaluadores](04_evaluadores_documentacion_tecnica.md) | #15, #09, #10 | RAGAS (6 métricas), bootstrap pareado, experimento de humanización, evaluador de seguridad SafeRAG. |

---

## Guías AESIA consideradas

| # | Título | Estado en este proyecto |
|---|--------|------------------------|
| 05 | Gestión de Riesgos | Parcialmente cubierta en doc 01 |
| 06 | Vigilancia Humana | Parcialmente cubierta en doc 01 |
| 07 | Datos y Gobernanza | Cubierta en doc 03 |
| 08 | Transparencia | Parcialmente cubierta en doc 01 |
| 09 | Precisión y Métricas | Cubierta en doc 04 |
| 10 | Solidez (Robustez) | Cubierta en doc 04 |
| 11 | Ciberseguridad | Parcialmente cubierta en doc 02 |
| 12 | Registros | Parcialmente cubierta en docs 02 y 04 |
| 13 | Vigilancia Poscomercialización | No aplicable — ver doc 00 |
| 14 | Gestión de Incidentes | No aplicable — ver doc 00 |
| 15 | Documentación Técnica | Base de todos los documentos |
| 16 | Declaración de Conformidad UE | No aplicable — ver doc 00 |
