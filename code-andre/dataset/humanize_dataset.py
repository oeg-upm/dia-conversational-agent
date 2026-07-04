"""
humanize_dataset.py
===================
Reescribe el campo `question` de cada entrada de
rag_dataset_v3_octen_qwen2.5_V2.json siguiendo los criterios de humanización:
  - 6-10 palabras (máx 12)
  - Sin estructuras formales de IA
  - Vocabulario coloquial, sin faltas
  - Misma intención semántica
  - Conserva referencias concretas
  - Adapta por tipo de pregunta

El resto de campos (ground_truth, contexts, metadata, etc.) no se tocan.
"""

import json
from pathlib import Path

SOURCE = Path("/Users/andrep/Documents/GitHub/dia-conversational-agent/code-juanma/dataset/rag_dataset_v3_octen_qwen2.5_V2.json")
OUTPUT = Path(__file__).parent / "rag_dataset_humanized_v1.json"

# ── Preguntas humanizadas (mismo orden que el dataset original) ───────────────
HUMANIZED = [
    # 1  comparative
    "¿En qué se diferencian S14 PLN y S15 Planificación automática?",
    # 2  ambiguous
    "¿Cuál es el cronograma de Sistemas de Planificación?",
    # 3  procedural
    "¿Cómo se trabaja la búsqueda informada en IA?",
    # 4  factual
    "¿Quién da Programación Declarativa en Matemáticas e Informática 2020-2021?",
    # 5  factual
    "¿Cuántos créditos tiene Aspectos Sociales/Legales/Éticos en el máster de Ciencia de Datos?",
    # 6  factual
    "¿Cuánto dura la actividad del tema 3.2 en Lingüística Computacional 2025-2026?",
    # 7  ambiguous
    "¿Para qué sirve Probabilidades y Estadística I en Matemáticas e Informática?",
    # 8  out_of_scope
    "¿Cuál es el horario de clases de Open Data and Knowledge Graphs?",
    # 9  factual
    "¿Qué pasa si la media ponderada es menor de 5 en el máster de Ingeniería Informática 2025-2026?",
    # 10 procedural
    "¿Cómo entrego los trabajos del tema 3 en Ciencia de la Web?",
    # 11 comparative
    "¿Cómo difiere la evaluación de UD1 frente a UD2 y UD3 en Probabilidades y Estadística I?",
    # 12 procedural
    "¿Cómo es la evaluación de Estadística en Ingeniería Informática y ADE 2023-2024?",
    # 13 procedural
    "¿Cómo se formalizan oraciones en Lógica de Primer Orden?",
    # 14 comparative
    "¿En qué difieren la evaluación ordinaria y la extraordinaria en Análisis Topológico de Datos 2025-2026?",
    # 15 comparative
    "¿Qué diferencia hay entre CE04 y CE09 en el máster de Biología Computacional?",
    # 16 ambiguous
    "¿Qué pasa si apruebo el examen pero suspendo las prácticas en Programación Declarativa?",
    # 17 ambiguous
    "¿Qué competencias necesito para trabajar en equipos interdisciplinarios en PLN?",
    # 18 procedural
    "¿Cómo modelo una solución con programación lógica en Programación Declarativa?",
    # 19 factual
    "¿Dónde se hacen las actividades de Ingeniería Ontológica en el máster IA?",
    # 20 comparative
    "¿Cómo se evalúan Principals of ML y AI for temporal data en Neurotechnology 2024-2025?",
    # 21 procedural
    "¿Cómo pido la evaluación extraordinaria en el máster IA 2025-2026?",
    # 22 factual
    "¿Cuántas horas de teoría hay en el Tema 1 de Programación Híbrida y Multiplataforma?",
    # 23 factual
    "¿De qué trata Análisis Topológico de Datos en CDIA 2025-2026?",
    # 24 factual
    "¿Cómo es la evaluación extraordinaria en Informática y ADE 2022-2023?",
    # 25 procedural
    "¿Cuándo se da IA y Ciencia Abierta en Ingeniería de Software Científico?",
    # 26 comparative
    "¿Qué diferencia hay entre aprendizaje automático y reconocimiento de patrones?",
    # 27 factual
    "¿Cuántos créditos tiene Metodología de Investigación en el máster de Ciencia de Datos?",
    # 28 factual
    "¿Qué dice la competencia CB04 sobre transmisión de información en PLN?",
    # 29 comparative
    "¿Qué diferencia hay entre las competencias y los resultados de aprendizaje en Ingeniería Ontológica?",
    # 30 factual
    "¿Cuánto duran las clases teóricas del tema 5 en Sistemas de Ayuda a la Decisión?",
    # 31 comparative
    "¿En qué difieren el test y las presentaciones orales en Redes Bayesianas?",
    # 32 out_of_scope
    "¿Quién da Lógica en Informática y ADE en 2023-2024?",
    # 33 factual
    "¿Qué se aprende al terminar Open Data y Knowledge Graphs en Innovación Digital?",
    # 34 factual
    "¿Cuánto dura cada lectura de artículos en Biología Programable?",
    # 35 factual
    "¿Qué aprendo sobre visualizaciones en Análisis de Grafos y Redes Sociales?",
    # 36 factual
    "¿Cuánto vale cada bloque en la nota de Lógica para IA?",
    # 37 factual
    "¿Qué nota mínima necesito en cada parcial de Lenguajes Formales 2021-2022?",
    # 38 procedural
    "¿Cómo elijo la evaluación por examen final en Deep Learning (itinerario Health)?",
    # 39 factual
    "¿Qué actividades de evaluación continua hay en Open Data and Knowledge Graphs (Health)?",
    # 40 factual
    "¿Qué asignaturas necesito aprobar antes de cursar Web Semántica y Linked Data?",
    # 41 factual
    "¿Cuántas horas de tutoría en grupo hay en Programación Lógica 2021-2022?",
    # 42 factual
    "¿Cuántos créditos tiene Ética y Ley para Ciencia de Datos?",
    # 43 comparative
    "¿En qué se diferencian CE14 y CE17 en CDIA?",
    # 44 comparative
    "¿En qué se diferencia el trabajo individual del grupal en IA en Neurotecnología?",
    # 45 ambiguous
    "¿Qué competencias cubre Lenguajes Formales en el doble grado 2025-2026?",
    # 46 procedural
    "¿Cómo es la convocatoria extraordinaria en Informática y ADE 2021-2022?",
    # 47 out_of_scope
    "¿Cuántas horas semanales tiene Minería de Datos en Ingeniería Informática?",
    # 48 comparative
    "¿Cómo difiere la evaluación ordinaria de la extraordinaria en Machine Learning?",
    # 49 comparative
    "¿Qué diferencia hay entre entrenar redes profundas y superficiales en Deep Learning?",
    # 50 procedural
    "¿Cómo se coordinan las actividades en Metodología de Investigación?",
    # 51 comparative
    "¿Qué diferencia hay entre las clases del tema 2.6 y el trabajo con exposición oral en Minería de Datos?",
    # 52 procedural
    "¿Cómo hago la primera parte del trabajo práctico en Estadística?",
    # 53 factual
    "¿Cómo se evalúa Sistemas de Planificación en Ingeniería Informática?",
    # 54 factual
    "¿En qué idioma se da Aspectos Legales, Sociales y Éticos de la IA?",
    # 55 comparative
    "¿En qué se diferencian RA146 y RA147 en la asignatura de Lógica?",
    # 56 out_of_scope
    "¿Qué competencias sobre IA en el sector legal cubre Derecho de Empresa?",
    # 57 procedural
    "¿Cómo se aprende en IA y Ciencia Abierta en Ingeniería de Software Científico?",
    # 58 factual
    "¿Cuántos créditos tiene Investigación Operativa en Ingeniería Informática?",
    # 59 ambiguous
    "¿Qué tiene que ver Lenguajes Formales con los ODS?",
    # 60 out_of_scope
    "¿Hay política de reembolso para los libros de Machine Learning?",
    # 61 procedural
    "¿Cómo consulto los horarios de tutoría de Deep Learning?",
    # 62 procedural
    "¿Cómo se evalúan las unidades 2 y 4 en Intelligent Systems?",
    # 63 comparative
    "¿En qué se diferencian CB01 y CB03 en CDIA?",
    # 64 procedural
    "¿Cómo me matriculo en Análisis Topológico de Datos Funcionales?",
    # 65 factual
    "¿Qué se aprende en Representación y Adquisición de Conocimiento en Biología Computacional?",
    # 66 factual
    "¿Cuántos créditos tiene Intelligent Systems en el máster de Innovación Digital?",
    # 67 comparative
    "¿Qué diferencia hay entre CE14 y CE17 en PLN de CDIA?",
    # 68 factual
    "¿Cuántas horas tiene Análisis Topológico de Series Espacio-Temporales en CDIA 2023-2024?",
    # 69 factual
    "¿En qué modalidad se imparte Programación Declarativa en Informática y ADE 2021-2022?",
    # 70 factual
    "¿Cuál es el email de Damiano Zanardini en Lógica 2020-2021?",
    # 71 factual
    "¿Qué dicen las competencias CG01 y CG02 en PLN de CDIA?",
    # 72 out_of_scope
    "¿Cuál es el horario de Lenguajes Formales en Informática y ADE 2020-2021?",
    # 73 procedural
    "¿Cómo me matriculo en Redes de Neuronas y Deep Learning del máster IA?",
    # 74 factual
    "¿Cuál es la bibliografía de Computación Social y Personalización 2023-2024?",
    # 75 procedural
    "¿Cómo está organizado el Tema 5 en Análisis Topológico de Datos Funcionales?",
    # 76 procedural
    "¿Cómo es el cronograma de Probabilidades y Estadística II 2021-2022?",
    # 77 comparative
    "¿Cómo es Sistemas Inteligentes comparado con otras asignaturas del máster en Ingeniería Informática?",
    # 78 procedural
    "¿Cómo consulto los horarios de tutoría en el máster de Innovación Digital 2020-2021?",
    # 79 procedural
    "¿Cómo veo los horarios de tutoría de Tecnologías Semánticas?",
    # 80 comparative
    "¿Cuántas horas de teoría hay frente a tutorías en Métodos de Simulación?",
    # 81 comparative
    "¿En qué se diferencian las dos evaluaciones de Lenguajes Formales en Matemáticas e Informática 2025-2026?",
    # 82 factual
    "¿Cuánto duran las presentaciones en Introducción a la Bioinformática?",
    # 83 factual
    "¿Cuál es el email del coordinador de Datos Abiertos y Grafos de Conocimiento?",
    # 84 comparative
    "¿Qué relación tienen CE14 y CE17 en PLN?",
    # 85 factual
    "¿Qué pasa con la nota teórica en la convocatoria ordinaria de CDIA 2022-2023?",
    # 86 factual
    "¿Cómo se llama la sección de datos descriptivos en la guía de Lingüística Computacional 2021-2022?",
    # 87 procedural
    "¿Cómo es la evaluación global de Representación del Conocimiento y Principios FAIR?",
    # 88 factual
    "¿Cuántos créditos tiene Minería de Datos en Ingeniería Informática?",
    # 89 factual
    "¿Cuál es el título completo de Fundamentos de Análisis de Imágenes en CDIA 2025-2026?",
    # 90 out_of_scope
    "¿Qué herramientas se usan para programación lineal en Decisiones Participativas y Negociación?",
    # 91 out_of_scope
    "¿Cuándo es la prueba de programación en Lógica?",
    # 92 ambiguous
    "¿Qué habilidades necesito para trabajar con grafos de conocimiento en Open Data?",
    # 93 comparative
    "¿En qué se diferencian las clases en aula y en laboratorio en Sistemas de Ayuda a la Decisión?",
    # 94 factual
    "¿Cuánto vale la modalidad OT en la nota de Probabilidades y Estadística I?",
    # 95 factual
    "¿Cuánto vale el examen práctico de la evaluación progresiva en Programación Declarativa?",
    # 96 comparative
    "¿En qué se diferencian las evaluaciones del tema 2 y el tema 4 en Information Retrieval (Health)?",
    # 97 comparative
    "¿En qué se diferencian el examen de UD1 y los de UD2 y UD3 en Probabilidades y Estadística II?",
    # 98 factual
    "¿Quién coordina Datos Abiertos y Grafos de Conocimiento en el máster Ingeniería Informática 2024-2025?",
    # 99 factual
    "¿Puedo entregar las prácticas aunque no haya entregado trabajos anteriores en Ciencia de Datos Health?",
    # 100 comparative
    "¿Qué relación hay entre los conocimientos previos y los resultados de aprendizaje en Programación Declarativa?",
]


def main():
    with open(SOURCE, encoding="utf-8") as f:
        items = json.load(f)

    assert len(items) == len(HUMANIZED), (
        f"Mismatch: dataset={len(items)} entries, HUMANIZED={len(HUMANIZED)} entries"
    )

    # Estadísticas
    word_counts = [len(q.split()) for q in HUMANIZED]
    changed = sum(1 for orig, new in zip(items, HUMANIZED) if orig["question"] != new)

    for item, new_q in zip(items, HUMANIZED):
        item["question"] = new_q
        item["generation_method"] = "humanized"   # marca la transformación

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Dataset guardado → {OUTPUT.name}")
    print(f"  Entradas:         {len(items)}")
    print(f"  Preguntas cambiadas: {changed} / {len(items)}")
    print(f"  Longitud media:   {sum(word_counts)/len(word_counts):.1f} palabras")
    print(f"  Longitud máxima:  {max(word_counts)} palabras")
    print(f"  Longitud mínima:  {min(word_counts)} palabras")
    print(f"  > 12 palabras:    {sum(1 for w in word_counts if w > 12)}")

    # Muestra 5 ejemplos aleatorios para revisión
    import random
    random.seed(42)
    print("\n── 5 ejemplos aleatorios ──")
    for i in random.sample(range(len(items)), 5):
        it = items[i]
        print(f"  [{i+1:3d}] [{it['question_type']:12s}] {it['question']}")


if __name__ == "__main__":
    main()
