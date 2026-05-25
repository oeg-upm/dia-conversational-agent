import pandas as pd

evaluations = [
    # --- MULTIQUERY ---
    {
        "input": "evaluation_results_octen_multiquery_deepseek_r1_distill_qwen_14b.csv",
        "output": "evaluation_results_octen_multiquery_deepseek_r1_distill_qwen_14b_mean.csv",
    },
    {
        "input": "evaluation_results_octen_multiquery_mistral_3_14b.csv",
        "output": "evaluation_results_octen_multiquery_mistral_3_14b_mean.csv",
    },
    {
        "input": "evaluation_results_octen_multiquery_nemotron3nano.csv",
        "output": "evaluation_results_octen_multiquery_nemotron3nano_mean.csv",
    },
    {
        "input": "evaluation_results_octen_multiquery_qwen2.5_32b.csv",
        "output": "evaluation_results_octen_multiquery_qwen2.5_32b_mean.csv",
    },
    {
        "input": "evaluation_results_octen_multiquery_qwen3_6_27b.csv",
        "output": "evaluation_results_octen_multiquery_qwen3_6_27b_mean.csv",
    },
    
    # --- NO MULTIQUERY ---
    {
        "input": "evaluation_results_octen_no_multiquery_deepseek_r1_distill_qwen_14b.csv",
        "output": "evaluation_results_octen_no_multiquery_deepseek_r1_distill_qwen_14b_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_multiquery_mistral_3_14b.csv",
        "output": "evaluation_results_octen_no_multiquery_mistral_3_14b_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_multiquery_nemotron3nano.csv",
        "output": "evaluation_results_octen_no_multiquery_nemotron3nano_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_multiquery_qwen2.5_32b.csv",
        "output": "evaluation_results_octen_no_multiquery_qwen2.5_32b_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_multiquery_qwen3_6_27b.csv",
        "output": "evaluation_results_octen_no_multiquery_qwen3_6_27b_mean.csv",
    }
]

metric_columns = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_similarity",
    "answer_correctness",
    "safe_behavior"
]

for evaluation in evaluations:

    print(f"Processing: {evaluation['input']}")

    df = pd.read_csv(evaluation["input"])

    mean_by_question = (
        df.groupby("question_type")[metric_columns]
        .mean()
        .round(5)
    )

    global_mean = (
        mean_by_question[metric_columns]
        .mean()
        .round(5)
    )

    mean_by_question.loc[""] = [""] * len(metric_columns)

    mean_by_question.loc["Global mean"] = global_mean

    mean_by_question.to_csv(evaluation["output"])

    print(f"Saved as: {evaluation['output']}")

print("All means have been calculated.")