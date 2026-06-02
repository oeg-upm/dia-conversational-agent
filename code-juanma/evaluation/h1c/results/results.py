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

    global_mean_row = df[df["question_type"] == "Global mean"]

    if global_mean_row.empty:
        print(f"Could not find 'Global mean' in {file}")
        continue

    row = global_mean_row.iloc[0]

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

    result = {
        "model": model,
        "prompt": prompt,
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

results = sorted(
    results,
    key=lambda x: (
        model_order[x["model"]],
        prompt_order[x["prompt"]]
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
                **{metric: "" for metric in metric_columns}
            }
        )

    final_rows.append(row)
    current_model = row["model"]

final_df = pd.DataFrame(final_rows)

final_df.to_csv("global_means_summary.csv", index=False)

print("File generated: global_means_summary.csv")
print(final_df)