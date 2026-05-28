import pandas as pd

evaluations = [
    {
        "input": "evaluation_results_bge.csv",
        "output": "evaluation_results_bge_mean.csv",
    },
    {
        "input": "evaluation_results_harrier.csv",
        "output": "evaluation_results_harrier_mean.csv",
    },
    {
        "input": "evaluation_results_octen.csv",
        "output": "evaluation_results_octen_mean.csv",
    },
    {
        "input": "evaluation_results_qwen8b.csv",
        "output": "evaluation_results_qwen8b_mean.csv",
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

    print(f"Procesando: {evaluation['input']}")

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