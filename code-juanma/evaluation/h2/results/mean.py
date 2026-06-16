import pandas as pd

evaluations = [
    {
        "input": "evaluation_results_octen_no_mq_deepseek_r1_distill_qwen_14b_few_shot.csv",
        "output": "evaluation_results_octen_no_mq_deepseek_r1_distill_qwen_14b_few_shot_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_mq_mistral_3_14b_reasoning_few_shot.csv",
        "output": "evaluation_results_octen_no_mq_mistral_3_14b_reasoning_few_shot_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_mq_nemotron3nano_few_shot.csv",
        "output": "evaluation_results_octen_no_mq_nemotron3nano_few_shot_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_mq_qwen_2.5_32b_few_shot.csv",
        "output": "evaluation_results_octen_no_mq_qwen_2.5_32b_few_shot_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_mq_qwen_3.6_27b_few_shot.csv",
        "output": "evaluation_results_octen_no_mq_qwen_3.6_27b_few_shot_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_mq_mistral_3_14b_instruct_few_shot.csv",
        "output": "evaluation_results_octen_no_mq_mistral_3_14b_instruct_few_shot_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_mq_gemma_3_12b_few_shot.csv",
        "output": "evaluation_results_octen_no_mq_gemma_3_12b_few_shot_mean.csv",
    },
    {
        "input": "evaluation_results_octen_no_mq_llama_3.1_8b_instruct_few_shot.csv",
        "output": "evaluation_results_octen_no_mq_llama_3.1_8b_instruct_few_shot_mean.csv",
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