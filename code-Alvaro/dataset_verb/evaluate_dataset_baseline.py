import json
import numpy as np
import pandas as pd
import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from ragas.run_config import RunConfig
import chromadb
custom_http_client = httpx.Client(timeout=1200.0)
custom_async_client = httpx.AsyncClient(timeout=1200.0)


ollama_url = "http://100.95.43.27:5000"

# RAG fijo — mismo para los dos experimentos
base_embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b",
    base_url=ollama_url
)
llm = ChatOpenAI(
        model="qwen2.5:32b",
        base_url="http://100.95.43.27:5000/v1",
        api_key="not_required",
        temperature=0.1,
        max_retries=2,
        http_client=custom_http_client,
        http_async_client=custom_async_client
)
vectorstore = Chroma(
    client=chromadb.HttpClient(host="localhost", port=8000),
    collection_name="basic_rag",
    embedding_function=base_embeddings
)

openai_client = OpenAI(
    base_url="http://100.95.43.27:5000/v1",
    api_key="not_required",
    timeout=450
)
ragas_llm = LangchainLLMWrapper(llm)

ragas_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

# Prevent VRAM overload by evaluating one item at a time
run_config = RunConfig(timeout=1200, max_workers=1)



RAG_PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente académico universitario. Responde usando el contexto proporcionado.
Si la información está presente aunque sea parcialmente, extráela y respóndela.
Solo di "No lo sé" si la información es completamente inexistente en el contexto.

REGLAS:
- Responde en español
- Responde de forma directa y concisa
- Incluye siempre los datos específicos: nombres, porcentajes, fechas, créditos
- No uses listas ni bullets

Contexto: {context}
Pregunta: {question}
Respuesta:""")

def run_evaluation(dataset_path: str, label: str):
    with open(dataset_path, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    #questions = questions[:20]  # Para pruebas rápidas, luego quitar este límite
    
    print(f"\n[{label}] Running {len(questions)} queries...")
    results = []
    # ← retriever movido dentro del bucle
    
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:70]}...")

        gt_c = q["ground_truth_context"].strip()

        # ← CAMBIO: retriever con filtro por source_document
        source = q.get("source_document", "")
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"source": source} if source else {}
            }
        )

        docs = retriever.invoke(q["question"])
        contexts = [d.page_content for d in docs]
        context_text = "\n\n".join(contexts)

        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": q["question"]})
        
        if not answer or len(answer.strip()) < 5:
            answer = "I don't know"

        print("\n====================")
        print("QUESTION:", q["question"])
        print("GROUND TRUTH ANSWER:", q["ground_truth"].strip())
        print("GROUND TRUTH Context:", gt_c)
        print("ANSWER:", answer)

        for j, c in enumerate(contexts):
            print(f"\nCONTEXT {j+1}:")
            print(c)
            

        results.append({
            "question": q["question"],
            "answer": answer.strip(),
            "contexts": [c.strip() for c in contexts],
            "ground_truth": q["ground_truth"].strip()
        })

    ragas_input = Dataset.from_list(results)
    
    scores = evaluate(
        ragas_input,
        metrics=[
            ContextRecall(),
            ContextPrecision(),
            Faithfulness(),
            AnswerRelevancy()
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        batch_size=2 
    ) 

    return {
        metric: np.mean(scores[metric]) if isinstance(scores[metric], list) else scores[metric]
        for metric in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    }

# Experimento A: evaluar con preguntas generadas sin verbalizar
# contra RAG indexado con documentos sin verbalizar
scores_baseline = run_evaluation("dataset_tables.json", "Baseline dataset")

print("\nRaw scores baseline:", scores_baseline)

metrics = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]

print("\n" + "="*50)
print(f"{'Métrica':<25} {'Baseline':>10}")
print("-"*50)

for m in metrics:
    a = scores_baseline[m]
    print(f"{m:<25} {a:>10.3f}")

summary_rows = []
for m in metrics:
    a = scores_baseline[m]
    summary_rows.append({
        "metric": m,
        "baseline": a,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("evaluation_summary_baseline_Table.csv", index=False, encoding="utf-8")
print("\nSummary saved to evaluation_summary_baseline_Table.csv")