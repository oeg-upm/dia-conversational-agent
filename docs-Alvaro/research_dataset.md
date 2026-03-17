# Generación del Dataset de Evaluación para el Sistema RAG

En este documento se presenta un análisis de las estrategias descritas en: Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems (https://arxiv.org/abs/2411.19710), para la generación del dataset de evaluación de nuestro sistema RAG del departamento de inteligencia artificial.

---

# Métodos de generación del dataset

Se contemplaron dos métodos de generación:

- **Statement Extraction Strategy**
- **Fine-tuning**

## Método Statement Extraction

Consiste en extraer un *statement* del documento y después generar una pregunta cuya respuesta sea esa afirmación. Este enfoque también se conoce como **answer-first generation**.

El proceso se divide en los siguientes pasos:

1. **Extraer statements del texto**.


2. **Generar preguntas** a partir de esos statements.

   Ejemplo de pregunta factual:  
   *¿Quién imparte la asignatura de Machine Learning en el Máster de Inteligencia Artificial?*

3. **Crear la tripleta**: question, answer, context


### Ventajas

- Garantiza que la respuesta esté en el documento, ya que el *statement* proviene directamente del texto.
- Permite que el **retriever** encuentre la información sin que el LLM tenga que inventarla.
- Reduce significativamente las **alucinaciones** del modelo.
- El paper muestra que con este método se obtiene un **dataset más balanceado** que con generación directa.
- Permite **controlar el tipo de preguntas** generadas mediante la taxonomía de preguntas.

### Desventajas

- El **pipeline es más largo**, debido al proceso de extracción de statements.
- Puede implicar un **mayor coste computacional**.

---

## Método Fine-Tuning

La idea principal de este método consiste en usar un **dataset existente de QA** para ajustar (*fine-tune*) el LLM generador de preguntas. Posteriormente, ese modelo se aplica a nuevos documentos para generar nuevas preguntas.

El modelo aprende la relación: context → question


### Problemas de este enfoque

- Tiende a generar **preguntas muy genéricas**, que no son adecuadas para evaluar un sistema RAG.
- Las preguntas **no siempre están alineadas con el texto**, pudiendo generar alucinaciones.
- **Requiere un dataset inicial** para el entrenamiento.

---

## Validación del dataset

Ambas estrategias requieren un **segundo paso de validación y limpieza** del dataset generado para garantizar su calidad.

---

# Tipos de preguntas del dataset

El paper contempla **tres categorías principales de preguntas**:

## 1. Factual

Requieren recuperar un **hecho específico** del corpus.

Ejemplo: ¿Quién imparte la asignatura de Machine Learning en el Máster de Inteligencia Artificial?


## 2. Summarization

Implican **sintetizar información** contenida en uno o varios fragmentos del documento.

## 3. Reasoning

Requieren **combinar múltiples afirmaciones del corpus** para inferir una respuesta.

---

# Diversidad del dataset

Es importante que el dataset **no esté dominado por un único tipo de preguntas**, ya que perderíamos la capacidad de evaluar distintas habilidades del sistema RAG.

Por ello se busca:

- Mantener **diversidad en el dataset**
- Incluir **distintos niveles de complejidad**
- Evaluar tanto recuperación directa como capacidades de síntesis y razonamiento

---

# Distribución de preguntas en nuestro proyecto

En nuestro proyecto:

- Las preguntas más comunes serán de tipo **factual**, debido a que el RAG contiene principalmente **información estructurada** sobre:
  - asignaturas
  - profesores
  - investigación
  - guías docentes

- Las preguntas de tipo **reasoning** también estarán presentes, pero **no dominarán el dataset**, ya que suelen requerir recuperar información de **múltiples fragmentos del corpus**, lo que aumenta su complejidad.

Se incluirá un número **reducido pero representativo** de estas preguntas para evaluar el comportamiento del sistema en escenarios más complejos.

### Ejemplo de distribución para nuestro proyecto 
- 50% factual
- 30% summarization
- 20% reasoning


---

# Evaluación del sistema RAG

Para analizar el rendimiento del sistema se puede observar **cómo responden el RAG y el LLM a los distintos tipos de preguntas**.

Por ejemplo:

- Si el sistema responde bien a preguntas **factual**
- Pero obtiene **bajo accuracy en preguntas reasoning**

Esto puede indicar problemas en:

- la **recuperación de chunks**
- el **razonamiento multifragmento**

El sistema RAG se podrá evaluar utilizando **RAGAS**, que permite medir el rendimiento del sistema a partir del dataset generado.

---

# Conclusión

Tras analizar las distintas estrategias descritas en  el paper se concluye que la mejor opción para generar el dataset de nuestro proyecto es **Statement Extraction Strategy**.

Este método permite:

- Garantizar que las respuestas de cada query estén **dentro del corpus**
- Reducir las **posibles alucinaciones del modelo**
- Controlar el **tipo de preguntas generadas**

Aunque implica un **pipeline más largo** y un **mayor coste computacional** que el enfoque basado en fine-tuning, ofrece **mayor control sobre la calidad del dataset**.

Además, el dataset deberá incluir **diferentes tipos de preguntas** para garantizar su diversidad:

- factual
- summarization
- reasoning

Con un **predominio de las preguntas factuales**, debido a la naturaleza informativa del contexto del RAG. Esta distribución permitirá evaluar el sistema en **distintos niveles de complejidad**.

