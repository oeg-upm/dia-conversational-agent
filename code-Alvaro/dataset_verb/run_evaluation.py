import numpy as np
import pandas as pd
from evaluate_dataset import run_evaluation  
import random

NUM_RUNS = 5  # Número de ejecuciones para evaluar la estabilidad de las métricas
DATASET_PATH = "dataset_tables.json"
QUESTION_TYPE = "" # Opcional: filtrar por tipo de pregunta Multi-hop Reasoning, Summarization, Factual, etc.
metrics = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]

all_runs = []

for run_id in range(1, NUM_RUNS + 1):
    print("\n" + "="*60)
    print(f"🚀 RUN {run_id}/{NUM_RUNS}")
    print("="*60)

    # 👇 opcional: cambiar seed en cada ejecución
    #random.seed(run_id)

    scores = run_evaluation(DATASET_PATH, f"Verbalizado run {run_id}", question_type="")

    print("\nRaw scores:", scores)

    # Guardar resultados individuales
    rows = []
    for m in metrics:
        rows.append({
            "run": run_id,
            "metric": m,
            "Verbalizado": scores[m]
        })

    df = pd.DataFrame(rows)
    filename = f"evaluation_verbalizado_{run_id}.csv"
    df.to_csv(filename, index=False, encoding="utf-8")

    print(f"✅ Saved: {filename}")

    all_runs.append(scores)

# ==========================================
# RESUMEN GLOBAL
# ==========================================

print("\n" + "="*60)
print("📊 RESUMEN GLOBAL")
print("="*60)

summary_rows = []

for m in metrics:
    values = [run[m] for run in all_runs]
    mean = np.mean(values)

    summary_rows.append({
        "metric": m,
        "Verbalizado": mean
    })

    print(f"{m:<25} mean={mean:.3f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("evaluation_summary_Verbalizado_table.csv", index=False, encoding="utf-8")

print("\n✅ Global summary saved to evaluation_summary_Verbalizado_table.csv")