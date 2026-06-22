"""
evaluate.py
------------------------

"""
import json
import httpx
import pandas
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

# Ragas wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


ollama_url = "http://100.86.34.41:11434"

custom_http_client = httpx.Client(timeout=1200.0)
custom_async_client = httpx.AsyncClient(timeout=1200.0)


# Base models
base_llm = ChatOpenAI(
    model="qwen2.5:32b", 
    base_url=f"{ollama_url}/v1",
    api_key="not_required",
    temperature=0,
    max_retries=2,
    http_client=custom_http_client,
    http_async_client=custom_async_client
)

base_embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b",
    base_url=ollama_url
)

# Wrappers config
evaluator_llm = LangchainLLMWrapper(base_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

# Prevent VRAM overload by evaluating one item at a time
run_config = RunConfig(timeout=1200, max_workers=1)

def run_evaluation(dataset_path: str, output_csv: str, sample_percentage: float = 0.5):
    """
    Evaluates a RAG dataset using Ragas metrics for H6 experiment (optimal k).
    
    Metrics:
      - Faithfulness:      detects hallucinations caused by irrelevant context (lost-in-the-middle).
      - AnswerRelevancy:   measures if the answer addresses the question despite noisy context.
      - ContextPrecision:  checks if the most relevant chunks are ranked at the top.
      - ContextRecall:     checks if all necessary information was retrieved.

    :param dataset_path: path to the JSON dataset.
    :param output_csv: path to save the evaluation results.
    :param sample_percentage: fraction of the dataset to evaluate (0.0 to 1.0).
    """
    try:
        print(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter out out_of_scope items (ground_truth label, not a real answer)
        #data = [item for item in data if item.get("ground_truth") != "out_of_scope"]

        total_items = len(data)
        num_samples = int(total_items * sample_percentage)

        if num_samples == 0 and sample_percentage > 0:
            num_samples = 1

        data = data[:num_samples]

        print(f"Total items in dataset (after filtering): {total_items}")
        print(f"Evaluating {num_samples} items ({sample_percentage * 100}% of the dataset)...")

        eval_dataset = Dataset.from_dict({
            "sample_id":          [item["sample_id"] for item in data],
            "question":           [item["question"] for item in data],
            "answer":             [item["answer"] for item in data],
            "contexts":           [item["contexts"] for item in data],
            "ground_truth":       [item["ground_truth"] for item in data],
            "reference_contexts": [item.get("reference_contexts", []) for item in data],
            "latency":            [item.get("latency", -1) for item in data],
        })

        # H6 metrics: retrieval quality + generation quality under varying k
        metrics = [
            Faithfulness(),      # generation: does more context introduce hallucinations?
            AnswerRelevancy(),   # generation: does more context hurt answer focus?
            ContextPrecision(),  # retrieval:  are relevant chunks ranked at the top?
            ContextRecall(),     # retrieval:  is all necessary info retrieved?
        ]

        print(f"\nStarting evaluation at {ollama_url}...")

        results = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
        )

        df_results = results.to_pandas()

        desired_columns = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        final_columns = [col for col in desired_columns if col in df_results.columns]
        df_filtered = df_results[final_columns].copy()
        df_filtered.insert(0, "sample_id", eval_dataset["sample_id"])
        df_filtered.insert(1, "generation_latency", eval_dataset["latency"])

        #df_filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")

        # Print mean scores for quick inspection
        print(f"\n=== EVALUATION COMPLETED ===")
        
        means = df_filtered[final_columns].mean()
        
        print(f"\n{'Metric':<25} {'Score':>8}")
        print("-" * 35)
        for metric, value in means.items():
            print(f"{metric:<25} {value:>8.4f}")
        print("-" * 35)
        print(f"{'mean_latency':<25} {df_filtered['generation_latency'].mean():>8.2f}s")
        print(f"\nResults saved to: {output_csv}")
        
        means_row = {col: round(df_filtered[col].mean(skipna=True), 4) for col in final_columns}
        means_row["sample_id"] = "MEAN"
        means_row["generation_latency"] = round(
            pandas.to_numeric(df_filtered["generation_latency"], errors="coerce")
            .replace(-1, pandas.NA).mean(skipna=True), 2
        )

        df_with_means = pandas.concat(
            [df_filtered, pandas.DataFrame([means_row])],
            ignore_index=True
        )

        df_with_means.to_csv(output_csv, index=False, encoding="utf-8-sig")

    except Exception as e:
        print(f"\nError during evaluation: {e}")


if __name__ == "__main__":
    run_evaluation(
        dataset_path="datasets/rag_dataset_v3_gemma4_26b.json",
        output_csv="evaluation_results.csv",
        sample_percentage=0.10
    )