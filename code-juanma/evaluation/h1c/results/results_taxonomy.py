import pandas as pd

files = [
    "evaluation_results_octen_no_mq_deepseek_r1_distill_qwen_14b_basic_mean.csv",
    "evaluation_results_octen_no_mq_deepseek_r1_distill_qwen_14b_structured_mean.csv",
    "evaluation_results_octen_no_mq_deepseek_r1_distill_qwen_14b_few_shot_mean.csv",
    "evaluation_results_octen_no_mq_qwen_2.5_32b_basic_mean.csv",
    "evaluation_results_octen_no_mq_qwen2.5_32b_structured_mean.csv",
    "evaluation_results_octen_no_mq_qwen_2.5_32b_few_shot_mean.csv",
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

results = []

for file in files:
    df = pd.read_csv(file)

    filename = file.replace("evaluation_results_", "").replace("_mean.csv", "")

    if "deepseek_r1_distill_qwen_14b" in filename:
        model = "deepseek-r1-distill-14b"
    else:
        model = "qwen2.5-32b"

    if filename.endswith("_basic"):
        prompt = "basic"
    elif filename.endswith("_structured"):
        prompt = "structured"
    elif filename.endswith("_few_shot"):
        prompt = "few_shot"
    else:
        prompt = ""

    df = df.dropna(subset=["question_type"])

    df = df[
        ~df["question_type"].isin(
            ["Global mean", ""]
        )
    ]

    for _, row in df.iterrows():

        result = {
            "model": model,
            "prompt": prompt,
            "taxonomy": row["question_type"]
        }

        for metric in metric_columns:
            result[metric] = row.get(metric)

        results.append(result)

prompt_order = {
    "basic": 0,
    "structured": 1,
    "few_shot": 2
}

model_order = {
    "deepseek-r1-distill-14b": 0,
    "qwen2.5-32b": 1
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
        prompt_order.get(x["prompt"], 999),
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
                "prompt": "",
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