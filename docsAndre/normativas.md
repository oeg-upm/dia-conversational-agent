Normativas Clave Aplicables
1. Reglamento (UE) 2024/1689 – Artificial Intelligence Act
Nombre oficial: Reglamento (UE) 2024/1689 por el que se establecen normas armonizadas en materia de Inteligencia Artificial
Fuente: Parlamento Europeo y Consejo de la Unión Europea
Publicación: 2024
Ámbito: Unión Europea
Relevancia para el proyecto
El sistema se clasifica como:
Sistema de riesgo limitado (Art. 50 – Transparencia).
No encaja en sistemas prohibidos (Art. 5).
No encaja en sistemas de alto riesgo (Art. 6 y Anexo III).
Obligaciones aplicables
Transparencia ante el usuario (Art. 50).
Identificación clara como sistema de IA.
No inducir a error respecto a naturaleza automatizada.
2. Reglamento (UE) 2016/679 – GDPR
Nombre oficial: Reglamento General de Protección de Datos
Fuente: Parlamento Europeo y Consejo
En vigor desde: 2018
Ámbito: Unión Europea
Relevancia
El sistema puede tratar:
Datos identificativos de profesores.
Información contenida en documentos PDF.
Logs de interacción.
Artículos clave aplicables
Art. 5 – Principios del tratamiento (minimización, limitación de finalidad).
Art. 6 – Base jurídica.
Arts. 13–14 – Transparencia.
Art. 35 – Evaluación de Impacto (DPIA).
Art. 5.2 – Responsabilidad proactiva (accountability).
3. Ley Orgánica 3/2018 (LOPDGDD)
Nombre oficial: Ley Orgánica de Protección de Datos Personales y garantía de los derechos digitales
Fuente: BOE – España
Año: 2018
Complementa el GDPR en el contexto español.
Relevante para:
Entidades públicas (universidades).
Derechos digitales.
Responsabilidad institucional.
4. Directiva (UE) 2022/2555 – NIS2
Nombre oficial: Directiva relativa a medidas destinadas a garantizar un elevado nivel común de ciberseguridad en la Unión
Fuente: Parlamento Europeo y Consejo
Año: 2022
Aplicable si el sistema se integra en infraestructura institucional.
Relevante para:
Gestión de riesgos de seguridad.
Protección de sistemas críticos.
Resiliencia operativa.
5. Real Decreto Legislativo 1/1996 – Ley de Propiedad Intelectual (España)
Fuente: BOE
Última actualización consolidada
Relevante porque el sistema utiliza:
Learning guides.
Documentación académica.
Debe garantizar:
Uso legítimo.
No redistribución indebida.
Respeto a derechos de autor.
6. OECD AI Principles (2019)
Fuente: Organización para la Cooperación y el Desarrollo Económicos (OCDE)
Principios:
IA centrada en el ser humano.
Robustez.
Transparencia.
Responsabilidad.
Marco ético complementario no vinculante pero altamente reconocido.
7. AESIA – Agencia Española de Supervisión de IA
Fuente: Gobierno de España
Periodo de referencia: 2024–2025
Relevante en:
Gobernanza de sistemas de IA.
Supervisión.
Trazabilidad.
Responsabilidad organizativa.
# Compliance Matrix – Agente Conversacional UPM

| Marco Normativo | Fuente | Artículo / Principio | Requisito | Impacto en el Sistema | Control Propuesto |
|-----------------|--------|---------------------|-----------|----------------------|------------------|
| Reglamento (UE) 2024/1689 (AI Act) | Parlamento Europeo y Consejo | Art. 50 – Transparencia | Informar que el usuario interactúa con un sistema de IA | Obligación de identificar el agente como IA | Disclaimer visible + etiqueta “Sistema de IA” |
| Reglamento (UE) 2024/1689 (AI Act) | Parlamento Europeo y Consejo | Art. 5 – Prácticas prohibidas | Prohibición de manipulación o inducción a error | El sistema no debe suplantar identidad humana | Diseño claro y no engañoso |
| Reglamento (UE) 2024/1689 (AI Act) | Parlamento Europeo y Consejo | Art. 9 – Gestión de riesgos (enfoque general) | Documentación y control de riesgos | Necesidad de gobernanza técnica | Registro de versiones + documentación técnica |
| Reglamento (UE) 2016/679 (GDPR) | Parlamento Europeo y Consejo | Art. 5.1.c – Minimización de datos | Tratamiento limitado a datos necesarios | Evitar almacenamiento excesivo | Indexación solo de documentos públicos |
| Reglamento (UE) 2016/679 (GDPR) | Parlamento Europeo y Consejo | Art. 6 – Base jurídica | Justificación legal del tratamiento | Uso bajo interés público institucional | Declaración de base jurídica |
| Reglamento (UE) 2016/679 (GDPR) | Parlamento Europeo y Consejo | Arts. 13–14 – Transparencia | Información clara sobre tratamiento de datos | Obligación de política de privacidad | Política accesible desde la interfaz |
| Reglamento (UE) 2016/679 (GDPR) | Parlamento Europeo y Consejo | Art. 35 – DPIA | Evaluación de impacto si procede | Análisis preventivo de riesgos | Mini DPIA documentada |
| Ley Orgánica 3/2018 (LOPDGDD) | BOE – España | Derechos digitales | Protección reforzada en contexto español | Supervisión institucional | Coordinación con DPO |
| Directiva (UE) 2022/2555 (NIS2) | Parlamento Europeo y Consejo | Gestión de riesgos de ciberseguridad | Protección frente a vulnerabilidades | Necesidad de medidas de seguridad | Logs + autenticación + control de acceso |
| Real Decreto Legislativo 1/1996 (LPI) | BOE – España | Derechos de reproducción | Uso legítimo de documentación académica | Evitar redistribución indebida | Uso exclusivo interno y sin descarga masiva |
| OECD AI Principles (2019) | OCDE | Human-centred AI | IA centrada en el usuario | Diseño responsable del sistema | Comunicación de limitaciones del sistema |
| Guías AESIA (2024–2025) | Gobierno de España | Supervisión humana y trazabilidad | Gobernanza organizativa del sistema | Necesidad de responsable designado | Modelo RACI + responsable institucional |


# Compliance Matrix V2 – Agente Conversacional UPM

| Nº | Marco Normativo / Guía | Referencia Jurídica | Criterio de Aplicabilidad | Riesgo Identificado | Obligación / Principio | Medida de Cumplimiento | Componente del Sistema |
|----|--------------------------|---------------------|---------------------------|--------------------|-----------------------|------------------------|------------------------|
| 1 | Reglamento (UE) 2024/1689 – AI Act | Art. 50.1 | El sistema interactúa directamente con personas físicas, ya sean estudiantes, profesores u otros. | El usuario puede desconocer que interactúa con un sistema automatizado | Transparencia obligatoria | Identificación explícita como sistema de IA en la interfaz (Un banner simple o integrado como texto fijo al final de cada respuesta)| UI (Gradio / Streamlit) |
| 2 | Reglamento (UE) 2024/1689 – AI Act | Art. 50.2 | El sistema genera contenido sintético | Confusión entre respuesta generada y comunicación institucional oficial | Transparencia sobre contenido generado | Aviso visible indicando que la respuesta es generada por IA | UI + RAG Backend (Generation) |
| 3 | Reglamento (UE) 2024/1689 – AI Act | Enfoque basado en riesgo (proporcionalidad) | Sistema clasificado como riesgo limitado | Generación de respuestas incorrectas o alucinadas | Mitigación razonable de riesgos previsibles | Arquitectura RAG restringida a corpus institucional con recuperación explícita de contexto (loop de evaluación con Nº 13)| RAG Backend (NN Search, Prompting, Generation) + VDB (Qdrant) |
| 4 | Reglamento (UE) 2016/679 – GDPR | Art. 5.1.c | Indexación de documentos con posibles datos personales | Tratamiento excesivo de datos personales | Principio de minimización | Validación previa del corpus antes de indexación | Parsing + Chunking + Indexing + Storage S3 |
| 5 | Reglamento (UE) 2016/679 – GDPR | Art. 6.1.e | Universidad pública en ejercicio de misión educativa | Tratamiento sin base jurídica válida | Base jurídica adecuada | Documentación del tratamiento bajo interés público | Documentación del Proyecto |
| 6 | Reglamento (UE) 2016/679 – GDPR | Arts. 13–14 | Posible registro de consultas y logs | Usuario desconoce el tratamiento de datos | Transparencia informativa | Política de privacidad accesible desde la interfaz | UI + API |
| 7 | Reglamento (UE) 2016/679 – GDPR | Art. 32 | Tratamiento electrónico de datos personales | Acceso no autorizado o pérdida de información | Seguridad del tratamiento | Control de acceso a API y base vectorial | API + VDB (Qdrant) + Storage S3 |
| 8 | Reglamento (UE) 2016/679 – GDPR | Art. 35 | Posible almacenamiento sistemático de interacciones | Riesgo para derechos y libertades | Evaluación de impacto cuando proceda | Mini DPIA documentada + limitación de retención de logs | API + Capa de Logs |
| 9 | Directiva (UE) 2022/2555 – NIS2 | Gestión de riesgos de ciberseguridad | Sistema desplegado en infraestructura institucional | Prompt injection o explotación de vulnerabilidades | Gestión estructurada de riesgos de seguridad | Validación de inputs y aislamiento de servicios (usar checklist 5 AESIA reducido) | API + RAG Backend + vLLM |
| 10 | Real Decreto Legislativo 1/1996 – LPI | Derechos de reproducción | Uso de learning guides protegidos | Redistribución indebida de material académico | Uso legítimo con fines educativos | No habilitar descarga automatizada ni exportación masiva | UI + Storage S3 |
| 11 | AESIA – Marco de Supervisión del AI Act (2024–2025) | Principio de responsabilidad organizativa | Sistema operado en entidad pública española | Ausencia de responsable identificable | Asignación formal de responsabilidad | Designación documentada de responsable del sistema, quien debería seguir un plan constante en base al checklist 6  | Gobernanza del Proyecto |
| 12 | AESIA – Criterios de Trazabilidad (alineados con AI Act) | Trazabilidad técnica | Imposibilidad de auditar funcionamiento | Falta de auditabilidad técnica | Registro y versionado técnico | Registro de versión de LLM, embeddings y corpus | vLLM + RAG Backend + Evaluation System |
| 13 | AESIA – Enfoque basado en riesgo | Evaluación proporcional al nivel de riesgo | Sistema clasificado como riesgo limitado | Falta de clasificación formal | Documentación del análisis de riesgo | Documento formal de clasificación como sistema de riesgo limitado (aqui falta profundizar como aplica la guía 5) | Documentación del Proyecto |
| 14 | OECD AI Principles (2019) | Robustez y seguridad | Respuestas desactualizadas o inconsistentes | Falta de fiabilidad informativa | Principio de robustez | Evaluación continua mediante RAGAS y DeepEval | Evaluation System |
| 15 | Reglamento (UE) 2024/1689 – AI Act | Requisitos de documentación técnica (gobernanza general) | Potencial supervisión futura | Falta de documentación estructurada | Documentación técnica mantenida y actualizada | Versionado de configuraciones del sistema RAG, aquí debemos incluir como checklist la guía 15 de la AESIA (que yo creo que será lo más extenso) | RAG System (Config 1, 2, 3) + Evaluation |
