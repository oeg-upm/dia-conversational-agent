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

# Global mean
data_extracted = []

for file in files:
    try:
        df = pd.read_csv(file)
        first_col = df.columns[0]
        global_mean_row = df[df[first_col] == "Global mean"]

        if global_mean_row.empty:
            print(f"Could not find 'Global mean' in {file}")
            continue

        row_dict = global_mean_row.iloc[0].to_dict()
        row_dict.pop(first_col, None)

        if "no_multiquery" in file:
            row_dict["multiquery"] = False
            row_dict["model_base"] = file.replace(
                "evaluation_results_octen_no_multiquery_", ""
            ).replace("_mean.csv", "")
        else:
            row_dict["multiquery"] = True
            row_dict["model_base"] = file.replace(
                "evaluation_results_octen_multiquery_", ""
            ).replace("_mean.csv", "")

        data_extracted.append(row_dict)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {file}. Ejecuta mean.py primero.")

df_all = pd.DataFrame(data_extracted)

# Pairwise comparison (Multiquery vs No Multiquery)
comparison_results = []

for model_name, group in df_all.groupby("model_base"):
    if len(group) < 2:
        print(f"Falta una de las versiones para el modelo: {model_name}")
        continue

    row_mq = group[group["multiquery"] == True].iloc[0]
    row_no_mq = group[group["multiquery"] == False].iloc[0]

    winner_row = {"model": model_name}

    for metric in metric_columns:
        val_mq = float(row_mq[metric])
        val_no_mq = float(row_no_mq[metric])

        multiplicador = 100 if (val_mq <= 1.0 and val_no_mq <= 1.0) else 1

        diff = abs(val_mq - val_no_mq) * multiplicador

        if val_mq > val_no_mq:
            winner_row[metric] = f"WITH_MQ (+{diff:.2f}%)"
        elif val_no_mq > val_mq:
            winner_row[metric] = f"NO_MQ (+{diff:.2f}%)"
        else:
            winner_row[metric] = "TIE (0.00%)"

    comparison_results.append(winner_row)

final_df = pd.DataFrame(comparison_results)
cols = ["model"] + metric_columns
final_df = final_df[cols]

# Global winner count
summary_row = {"model": "TOTAL_WITH_MQ_WINS"}
summary_row_no = {"model": "TOTAL_NO_MQ_WINS"}

for metric in metric_columns:
    summary_row[metric] = final_df[metric].str.contains("WITH_MQ").sum()
    summary_row_no[metric] = final_df[metric].str.contains("NO_MQ").sum()

empty_row = {col: "" for col in final_df.columns}
final_df.loc[len(final_df)] = empty_row
final_df.loc[len(final_df)] = summary_row
final_df.loc[len(final_df)] = summary_row_no

output_filename = "multiquery_vs_no_multiquery_winners.csv"
final_df.to_csv(output_filename, index=False)

print(f"\nGenerated file: {output_filename}")
print(final_df.to_string())