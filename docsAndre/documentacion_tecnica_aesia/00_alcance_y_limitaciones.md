# Alcance y Limitaciones de la Documentación Técnica
## Proyecto DIA — Agente Conversacional RAG para la FI-UPM

**Guías de referencia principales:** AESIA #15 (Documentación Técnica), con elementos de guías #05–#12  
**Fecha de redacción:** Julio 2026  
**Contexto:** Proyecto académico de investigación, Escuela Técnica Superior de Ingeniería Informática, UPM

---

## 1. Propósito de este documento

Este documento sirve como declaración explícita de qué secciones de las guías AESIA son:

- **Aplicables y cumplidas** en su totalidad
- **Parcialmente aplicables**: el proyecto no cumple el requisito al 100%, pero existe un artefacto concreto que cubre el espíritu del requisito
- **No aplicables** por naturaleza del contexto académico, con justificación

El objetivo es ser máximamente transparente, aplicando el criterio de "apertura razonada": si existe cualquier aspecto del proyecto que se pueda relacionar con una sección regulatoria, se señala, aunque la cobertura sea parcial.

---

## 2. Resumen de guías AESIA consideradas

| # | Título | Aplicabilidad al proyecto |
|---|--------|--------------------------|
| 05 | Gestión de riesgos | Parcial — riesgos identificados informalmente |
| 06 | Vigilancia humana | Parcial — arquitectura requiere selección manual de contexto |
| 07 | Datos y gobernanza | Aplicable — corpus y datasets documentados |
| 08 | Transparencia e información | Parcial — sistema se identifica como IA en el prompt |
| 09 | Precisión y métricas | Aplicable — RAGAS + bootstrap estadístico |
| 10 | Solidez (robustness) | Aplicable — experimento de humanización |
| 11 | Ciberseguridad | Parcial — VPN Tailscale, API expuesta |
| 12 | Registros (logging) | Aplicable — logs de backend existentes |
| 13 | Vigilancia poscomercialización | No aplicable — proyecto académico, sin despliegue en producción |
| 14 | Gestión de incidentes | Parcial — incidentes técnicos documentados informalmente |
| 15 | Documentación técnica | Aplicable — este conjunto de documentos |
| 16 | Declaración de conformidad UE | No aplicable — ver §3 |

---

## 3. Secciones no aplicables por contexto académico

### 3.1 Declaración de conformidad UE (Guía 16 / Anexo IV, Art. 47-49)

**Requisito formal:** El proveedor del sistema de IA debe emitir una declaración de conformidad firmada antes de comercializar o poner en servicio el sistema.

**Situación del proyecto:** No existe proveedor comercial. El proyecto es una prueba de concepto académica desarrollada en el contexto de un trabajo de fin de grado/máster. No hay acto de "puesta en servicio" en el sentido del Reglamento Europeo de IA.

**Conexión parcial real:** La documentación técnica que se genera en estos documentos (`01`–`04`) puede considerarse el precursor de una declaración de conformidad; cubre el contenido técnico que dicha declaración requeriría (descripción del sistema, métricas de evaluación, análisis de riesgos).

---

### 3.2 Vigilancia poscomercialización (Guía 13 / Art. 72-74)

**Requisito formal:** Sistema de vigilancia activa en producción, informes periódicos a autoridad de supervisión, plazos de conservación de incidentes (hasta 10 años).

**Situación del proyecto:** El sistema no está desplegado en producción. Los experimentos se ejecutan en un clúster universitario de acceso privado (Tailscale VPN). No existe base de usuarios reales ni ciclo de vida poscomercialización.

**Conexión parcial real:** Los archivos de log generados por el backend (`backend_{id}.log`) constituyen la base de lo que sería un sistema de vigilancia. La metodología de evaluación RAGAS aplicada periódicamente (tras cada cambio de configuración) sigue el espíritu de monitorización continua de rendimiento.

---

### 3.3 Conservación de registros a 10 años (Guía 12, §4.4)

**Requisito formal:** El proveedor debe conservar los registros técnicos durante al menos 10 años tras la retirada del sistema del mercado.

**Situación del proyecto:** No existe infraestructura de retención a largo plazo. El clúster universitario con GPUs NVIDIA H100 es de acceso temporal.

**Conexión parcial real:** Los resultados de evaluación (CSV), los datasets generados (JSON) y el código fuente en el repositorio Git constituyen el registro técnico factual del proyecto. Git actúa como sistema de trazabilidad de versiones, cumpliendo el espíritu de conservación de historial de cambios.

---

### 3.4 Organismos notificados y auditoría externa (Anexo IV, Art. 43)

**Requisito formal:** Para sistemas de IA de alto riesgo, una evaluación de conformidad por un organismo notificado.

**Situación del proyecto:** El sistema no está categorizado formalmente como sistema de alto riesgo en el sentido del Reglamento. Es un prototipo académico de asistente de consulta de guías docentes universitarias, sin toma de decisiones de alto impacto sobre personas.

**Conexión parcial real:** La evaluación de seguridad (dataset de 60 prompts adversariales siguiendo metodología SafeRAG) y la evaluación RAGAS pueden considerarse análogas a una auditoría técnica interna.

---

### 3.5 Gestión de incidentes grave (Guía 14 / Art. 73)

**Requisito formal:** Procedimientos formalizados para notificar incidentes graves a la autoridad nacional de supervisión de IA en plazo de 15 días.

**Situación del proyecto:** No existe interacción con usuarios reales, por lo que el riesgo de incidente grave (daño a persona) es inexistente en el contexto actual.

**Conexión parcial real:** Durante el desarrollo se documentaron incidentes técnicos (errores CUDA, timeouts de Ollama, errores SSL en macOS con fork). Estas incidencias, registradas en el historial Git y discutidas en la documentación del proyecto, constituyen un precedente informal de gestión de incidentes técnicos.

---

## 4. Criterios de clasificación de riesgo del sistema

De acuerdo con el Reglamento Europeo de IA (EU AI Act, Art. 6 y Anexo III), el sistema DIA se clasifica como:

- **Categoría:** Sistema de IA de propósito general de bajo riesgo (no incluido en Anexo III de alto riesgo)
- **Justificación:** No toma decisiones sobre admisión académica, calificaciones, selección de personal ni ninguna otra categoría de alto riesgo. Es un asistente de consulta de información pública (guías docentes) para estudiantes universitarios.
- **Implicación:** No está sujeto a los requisitos estrictos del Capítulo III del Reglamento. Sin embargo, sí aplican las obligaciones generales de transparencia (Art. 50) al ser un sistema que interactúa con personas físicas mediante IA generativa.

---

## 5. Decisiones de diseño adoptadas con perspectiva regulatoria

Las siguientes decisiones de diseño del proyecto, aunque tomadas por razones técnicas, están alineadas con los principios regulatorios de las guías AESIA:

| Decisión de diseño | Principio regulatorio relacionado |
|--------------------|----------------------------------|
| El sistema requiere que el usuario seleccione explícitamente los documentos de contexto antes de preguntar | Guía 06: Vigilancia humana — el usuario mantiene control sobre el alcance de la consulta |
| El prompt del sistema prohíbe explícitamente al LLM fabricar información no presente en el contexto | Guía 08: Transparencia — el sistema no simula omnisciencia |
| El sistema detecta preguntas `out_of_scope` y `ambiguous` y responde de forma diferenciada | Guía 06: Vigilancia humana — el sistema no actúa de forma autónoma cuando no tiene certeza |
| La evaluación incluye prompts adversariales de inyección de prompt (prompt injection) | Guía 11: Ciberseguridad — prueba activa de vectores de ataque |
| Toda la infraestructura del clúster es accesible únicamente mediante VPN (Tailscale) | Guía 11: Ciberseguridad — aislamiento de red |
| Los resultados de evaluación se calculan con intervalos de confianza estadísticos (bootstrap pareado N=2000) | Guía 09: Precisión — significancia estadística de las métricas de rendimiento |
| El experimento de humanización evalúa el sistema bajo variación de estilo de input | Guía 10: Solidez — evaluación formal de robustez ante perturbaciones de entrada |

---

*Este documento forma parte del conjunto de documentación técnica del proyecto DIA. Los demás documentos se enumeran en el [índice de documentación](README.md).*
