import pandas as pd

files = [
    "evaluation_results_octen_no_mq_deepseek_r1_distill_qwen_14b_few_shot_mean.csv",
    "evaluation_results_octen_no_mq_mistral_3_14b_reasoning_few_shot_mean.csv",
    "evaluation_results_octen_no_mq_mistral_3_14b_instruct_few_shot_mean.csv",
    "evaluation_results_octen_no_mq_nemotron3nano_few_shot_mean.csv",
    "evaluation_results_octen_no_mq_qwen_2.5_32b_few_shot_mean.csv",
    "evaluation_results_octen_no_mq_qwen_3.6_27b_few_shot_mean.csv",
    "evaluation_results_octen_no_mq_gemma_3_12b_few_shot_mean.csv",
    "evaluation_results_octen_no_mq_llama_3.1_8b_instruct_few_shot_mean.csv",
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

model_mapping = {
    "deepseek_r1_distill_qwen_14b": "deepseek-r1-distill-14b",
    "mistral_3_14b_instruct": "mistral-3-14b-instruct",
    "nemotron3nano": "nemotron3nano",
    "qwen_2.5_32b": "qwen-2.5-32b",
    "qwen_3.6_27b": "qwen-3.6-27b",
    "gemma_3_12b": "gemma-3-12b",
    "llama_3.1_8b_instruct": "llama-3.1-8b-instruct",
    "mistral_3_14b_reasoning": "mistral-3-14b-reasoning",
}

results = []

for file in files:
    df = pd.read_csv(file)

    filename = file.replace("evaluation_results_", "").replace("_mean.csv", "")

    model = None

    for key, value in model_mapping.items():
        if key in filename:
            model = value
            break

    if model is None:
        model = filename

    df = df.dropna(subset=["question_type"])

    df = df[
        ~df["question_type"].isin(
            ["Global mean", ""]
        )
    ]

    for _, row in df.iterrows():
        result = {
            "model": model,
            "taxonomy": row["question_type"]
        }

        for metric in metric_columns:
            result[metric] = row.get(metric)

        results.append(result)

model_order = {
    "deepseek-r1-distill-14b": 0,
    "mistral-3-14b-reasoning": 1,
    "mistral-3-14b-instruct": 2,
    "nemotron3nano": 3,
    "qwen-2.5-32b": 4,
    "qwen-3.6-27b": 5,
    "gemma-3-12b": 6,
    "llama-3.1-8b-instruct": 7,
    
}

taxonomy_order = {
    "ambiguous": 0,
    "comparative": 1,
    "factual": 2,
    "out_of_scope": 3,
    "procedural": 4
}

results = sorted(
    results,
    key=lambda x: (
        model_order.get(x["model"], 999),
        taxonomy_order.get(x["taxonomy"], 999)
    )
)

final_rows = []

current_model = None

for row in results:
    if current_model is not None and row["model"] != current_model:
        final_rows.append(
            {
                "model": "",
                "taxonomy": "",
                **{metric: "" for metric in metric_columns}
            }
        )

    final_rows.append(row)
    current_model = row["model"]

final_df = pd.DataFrame(final_rows)

final_df.to_csv("taxonomy_summary.csv", index=False)

print("File generated: taxonomy_summary.csv")
print(final_df)