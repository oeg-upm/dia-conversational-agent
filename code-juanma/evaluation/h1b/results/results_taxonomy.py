import pandas as pd

files = [
    # --- MULTIQUERY ---
    "evaluation_results_octen_multiquery_deepseek_r1_distill_qwen_14b_mean.csv",
    "evaluation_results_octen_multiquery_mistral_3_14b_mean.csv",
    "evaluation_results_octen_multiquery_nemotron3nano_mean.csv",
    "evaluation_results_octen_multiquery_qwen2.5_32b_mean.csv",
    "evaluation_results_octen_multiquery_qwen3.14b_mean.csv",
    "evaluation_results_octen_multiquery_qwen3_6_27b_mean.csv",

    # --- NO MULTIQUERY ---
    "evaluation_results_octen_no_multiquery_deepseek_r1_distill_qwen_14b_mean.csv",
    "evaluation_results_octen_no_multiquery_mistral_3_14b_mean.csv",
    "evaluation_results_octen_no_multiquery_nemotron3nano_mean.csv",
    "evaluation_results_octen_no_multiquery_qwen2.5_32b_mean.csv",
    "evaluation_results_octen_no_multiquery_qwen3.14b_mean.csv",
    "evaluation_results_octen_no_multiquery_qwen3_6_27b_mean.csv",
]

metric_columns = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_similarity",
    "answer_correctness",
    "safe_behavior",
]

all_rows = []

for file in files:
    try:
        df = pd.read_csv(file)
        taxonomy_col = df.columns[0]

        df = df[
            df[taxonomy_col].astype(str).str.strip().str.lower()
            != "global mean"
        ]

        if "no_multiquery" in file:
            model_name = (
                file.replace("evaluation_results_octen_no_multiquery_", "")
                .replace("_mean.csv", "")
                + "_NO_MQ"
            )
        else:
            model_name = (
                file.replace("evaluation_results_octen_multiquery_", "")
                .replace("_mean.csv", "")
                + "_MQ"
            )

        for _, row in df.iterrows():
            all_rows.append({
                "Model": model_name,
                "Taxonomy": row[taxonomy_col],
                **{m: row[m] if m in df.columns else None for m in metric_columns}
            })

    except FileNotFoundError:
        print(f"File not found: {file}")

df = pd.DataFrame(all_rows)

df = df.sort_values(
    by=["Model", "Taxonomy"],
    kind="stable"
).reset_index(drop=True)

final_rows = []
prev_model = None

for _, row in df.iterrows():

    if prev_model is not None and row["Model"] != prev_model:
        final_rows.append({
            "Model": None,
            "Taxonomy": None,
            **{m: None for m in metric_columns}
        })

    final_rows.append(row.to_dict())
    prev_model = row["Model"]

final_df = pd.DataFrame(final_rows)

final_df = final_df[
    ~(
        final_df["Model"].notna() &
        final_df["Taxonomy"].isna()
    )
]

# -------------------------
# SAVE
# -------------------------
output_file = "all_models_by_taxonomy.csv"
final_df.to_csv(output_file, index=False)

print(f"\nGenerated: {output_file}")
print(final_df.to_string())