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

# Legacy imports
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity 
)


ollama_url = "http://100.71.243.90:5000"

custom_http_client = httpx.Client(timeout=1200.0)
custom_async_client = httpx.AsyncClient(timeout=1200.0)


# Base models
base_llm = ChatOpenAI(
    model="qwen3.5:35b", 
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

def run_evaluation(dataset_path: str, output_csv: str, sample_percentage: float = 1.0):
    """
    Evaluates a RAG dataset using Ragas metrics.
    
    :param dataset_path: path to the JSON dataset.
    :param output_csv: path to save the evaluation results.
    :param sample_percentage: fraction of the dataset to evaluate (0.0 to 1.0).
    """
    try:
        print(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        total_items = len(data)
        num_samples = int(total_items * sample_percentage)
        
        if num_samples == 0 and sample_percentage > 0:
            num_samples = 1
            
        data = data[:num_samples]
        
        print(f"Total items in dataset: {total_items}")
        print(f"Evaluating {num_samples} items ({sample_percentage * 100}% of the dataset)...")
        
        # Prepare HuggingFace dataset format including metadata
        eval_dataset = Dataset.from_dict({
            "sample_id": [item["sample_id"] for item in data],
            "question": [item["question"] for item in data],
            "answer": [item["answer"] for item in data],
            "contexts": [item["contexts"] for item in data],
            "ground_truth": [item["ground_truth"] for item in data],
            "reference_contexts": [item.get("reference_contexts", []) for item in data]
        })

        # Initialize metrics
        metrics = [
            Faithfulness(),       # Measures if the answer is strictly based on the context (detects hallucinations).
            AnswerRelevancy(),    # Evaluates how directly the answer addresses the question, penalizing fluff.
            ContextPrecision(),   # Assesses if the most relevant retrieved documents are ranked at the top.
            ContextRecall(),      # Checks if the retriever found all the necessary information to fully answer.
            AnswerSimilarity(),   # Computes the semantic similarity between the generated answer and the ground truth.
            AnswerCorrectness()   # Measures factual accuracy by comparing the generated answer against the ground truth.
        ]

        print(f"\nStarting evaluation at {ollama_url}...")
        
        # Execute Ragas evaluation
        results = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config
        )

        df_results = results.to_pandas()

        desired_columns = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_similarity",
            "answer_correctness"
        ]
        
        # Filter columns to only include those that are present in the results
        final_columns = [col for col in desired_columns if col in df_results.columns]
        df_filtered = df_results[final_columns]

        # Add sample_id back to the DataFrame for reference
        df_filtered.insert(0, "sample_id", eval_dataset["sample_id"])

        df_filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")
        
        print(f"\n=== EVALUATION COMPLETED ===")
        print(f"Results successfully saved to: {output_csv}")

    except Exception as e:
        print(f"\nError during evaluation: {e}")

if __name__ == "__main__":

    run_evaluation(
        dataset_path="datasets/rag_dataset_v3_gemma4_26b.json",
        output_csv="evaluation_results.csv",
        sample_percentage=0.10
    )