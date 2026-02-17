# Guía rápida de consultoría (scope & requisitos)

## 1) Stakeholders y objetivos
- ¿Quién usa el bot? (alumno, profesor, PAS, coordinación)
- ¿Qué problemas concretos resuelve? (ej. "cómo se evalúa X", "requisitos", "bibliografía")
- ¿Qué preguntas NO debe contestar? (límites explícitos)

## 2) Fuentes de verdad (ground truth)
- Lista de PDFs oficiales (learning guides) + versiones por año
- Fuentes de profes (web oficial / directorio)
- Políticas/FAQ institucionales (si aplica)
- Reglas: solo responder con evidencia citada (doc + página/sección)

## 3) Datos sensibles / privacidad
- ¿Se permite info de alumnos pasados? (en principio NO / solo agregada o anonimizada)
- ¿Se guardan logs de conversaciones? ¿cuánto tiempo? ¿anonimización?

## 4) Taxonomía de intents (mínimo)
- Curso: evaluación, requisitos, temario, bibliografía, competencias, calendario
- Profesor: contacto, tutorías, asignaturas impartidas (solo si es público/oficial)
- Procedimientos: matrícula, cambios, reclamaciones (solo fuentes oficiales)

## 5) Formato de respuesta
- Respuesta breve + citas
- “No encontrado” + petición de aclaración (curso/año/idioma)
- Link a sección relevante del PDF si es posible

## 6) Evaluación
- Set de Q&A “gold” por curso (20–50)
- Métricas: % respuestas con citas correctas, recall@k del retrieval, tasa de “no sé” razonable
