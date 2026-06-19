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

results = []

for file in files:
    df = pd.read_csv(file)
    global_mean_row = df[df["question_type"] == "Global mean"]

    if global_mean_row.empty:
        print(f"Could not find 'Global mean' in {file}")
        continue

    row_dict = global_mean_row.iloc[0].to_dict()
    row_dict.pop("question_type", None)

    row_dict["model"] = (
        file.replace("_mean.csv", "")
            .replace("evaluation_results_", "")
    )

    results.append(row_dict)

final_df = pd.DataFrame(results)

cols = ["model"] + metric_columns
final_df = final_df[cols]

# =========================
# WINNER PER METRIC
# =========================

winner_row = {"model": "WINNER"}

for metric in metric_columns:

    max_idx = final_df[metric].astype(float).idxmax()
    winner_model = final_df.loc[max_idx, "model"]
    winner_row[metric] = winner_model
empty_row = {col: "" for col in final_df.columns}

final_df.loc[len(final_df)] = empty_row
final_df.loc[len(final_df)] = winner_row

final_df.to_csv("global_means_summary.csv", index=False)

print("File generated: global_means_summary.csv")
print(final_df)