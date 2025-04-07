import os
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datasets import load_from_disk
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import pathlib

import utils
from options import Config

matplotlib.use("Agg")


def load_model():
    """
    Load the pre-trained RoBERTa model fine-tuned on STS benchmark (for semantic similarity).

    Returns:
        SentenceTransformer: Pretrained model for embedding generation.
    """
    return SentenceTransformer("roberta-large-nli-stsb-mean-tokens")


def get_embedding(text, model):
    """
    Generate embedding for a given text using the RoBERTa model.

    Args:
        text (str): Input text.
        model (SentenceTransformer): Loaded embedding model.

    Returns:
        Tensor: Embedding tensor.
    """
    return model.encode(text, convert_to_tensor=True)


def compute_similarity(correct_answer, distractors, model):
    """
    Compute cosine similarity between correct answer and distractors.
    While replacing any missing values in the distractors with 'empty'

    Args:
        correct_answer (str): The correct answer text.
        distractors (list of str): List of distractor texts.
        model (SentenceTransformer): The model to generate embeddings.

    Returns:
        Cosine similarity scores for each distractor.
    """
    # Handle every empty value found for distractors
    cleaned_distractors = [
        d if d is not None and d.strip() != "" else "empty" for d in distractors
    ]

    num_filled = sum(1 for d in distractors if not d or d.strip() == "")
    if num_filled > 0:
        print(
            f"Replaced {num_filled} empty distractor(s) with 'empty' for question: {correct_answer}"
        )

    # Compute cosine similarity
    sentences = [correct_answer] + cleaned_distractors
    embeddings = model.encode(sentences, convert_to_tensor=True)

    correct_emb = embeddings[0]
    distractor_embs = embeddings[1:]

    similarity_scores = [
        util.pytorch_cos_sim(correct_emb, emb).item() for emb in distractor_embs
    ]
    return similarity_scores


def model_probabilities(questions, correct_answers, distractors_list, qa_pipeline):
    """
    Use a QA pipeline to get the model's confidence score for each answer option.
    Also handel the empty values found for the distractors.

    Args:
        questions (list of str): List of questions.
        correct_answers (list of str): List of correct answers.
        distractors_list (list of list of str): List of distractors for each question.
        qa_pipeline: Question Answering pipeline.

    Returns:
        list of list of dictionaries: Scores and predicted answers for each option per question
        are returned.
    """
    all_results = []

    for q, correct, distractors in zip(questions, correct_answers, distractors_list):
        # Replace empty distractors with 'empty'
        distractors = [
            d if isinstance(d, str) and d.strip() else "empty" for d in distractors
        ]
        options = [correct] + distractors
        qa_inputs = [{"question": q, "context": opt} for opt in options]

        # Filter out invalid (empty) inputs
        valid_inputs = [
            entry
            for entry in qa_inputs
            if entry["question"].strip() and entry["context"].strip()
        ]

        try:
            results = qa_pipeline(valid_inputs) if valid_inputs else []
        except Exception:
            results = []

        # Match scores back to the original options
        result_iter = iter(results)
        full_result = []
        for entry in qa_inputs:
            if entry["question"].strip() and entry["context"].strip():
                result = next(result_iter, {"score": 0.0})
                full_result.append(
                    {
                        "context": entry["context"],
                        "score": result["score"],
                        "answer": result.get("answer", ""),
                    }
                )
            else:
                full_result.append(
                    {"context": entry["context"], "score": 0.0, "answer": ""}
                )
        all_results.append(full_result)

    return all_results


def jaccard_similarity(str1, str2):
    """
    Calculate Jaccard similarity between two strings based on word overlap.
    """
    words1 = set(str1.split())
    words2 = set(str2.split())
    return len(words1 & words2) / len(words1 | words2)


def compute_lexical_similarity_jaccard(correct_answer, distractors):
    """
    Compute Jaccard similarity between correct answer and each distractor.
    Here also handle empty distractores the same way as the previous two
    funtions.
    """
    # Replace empty distractors with 'empty'
    distractors = [
        d if isinstance(d, str) and d.strip() else "empty" for d in distractors
    ]

    # Compute lexical similarity
    return [jaccard_similarity(correct_answer, d) for d in distractors]


def process_dataset(dataset, model, qa_model):
    """
    Main processing function to compute all similarity and probability metrics.

    Args:
        dataset: Dataset with questions, correct answers, and distractors.
        Where the distractores are generated by the models in this project
        model: Semantic similarity model.
        qa_model: QA pipeline for probability computation.

    Returns:
        df_results (DataFrame): Full metrics dataframe.
        summary_stats (dict): Basic descriptive statistics.
    """
    # Get general information for DataFrame
    questions = [row["question"] for row in dataset]
    supports = [row["support"] for row in dataset]
    correct_answers = [row["correct_answer"] for row in dataset]
    distractors_list = [
        [row["distractor1"], row["distractor2"], row["distractor3"]] for row in dataset
    ]

    # Compute semantic similarity scores
    similarity_scores_all = [
        compute_similarity(correct, distractors, model)
        for correct, distractors in zip(correct_answers, distractors_list)
    ]
    similarity_scores1 = [s[0] for s in similarity_scores_all]
    similarity_scores2 = [s[1] for s in similarity_scores_all]
    similarity_scores3 = [s[2] for s in similarity_scores_all]

    # Compute probabilities for distractors and the good answers
    probabilities = model_probabilities(
        questions, correct_answers, distractors_list, qa_model
    )
    good_answer_score = [
        probabilities[i][0]["score"] for i in range(len(probabilities))
    ]
    probability_score1 = [
        probabilities[i][1]["score"] for i in range(len(probabilities))
    ]
    probability_score2 = [
        probabilities[i][2]["score"] for i in range(len(probabilities))
    ]
    probability_score3 = [
        probabilities[i][3]["score"] for i in range(len(probabilities))
    ]

    # Compute lexical similarity scores
    lexical_similarity_all = [
        compute_lexical_similarity_jaccard(correct, distractors)
        for correct, distractors in zip(correct_answers, distractors_list)
    ]
    lexical_similarity_scores1 = [s[0] for s in lexical_similarity_all]
    lexical_similarity_scores2 = [s[1] for s in lexical_similarity_all]
    lexical_similarity_scores3 = [s[2] for s in lexical_similarity_all]

    # Create DataFrame with all scores and information needed
    df_results = pd.DataFrame(
        {
            "question": questions,
            "support": supports,
            "correct_answer": correct_answers,
            "distractor1": [d[0] for d in distractors_list],
            "distractor2": [d[1] for d in distractors_list],
            "distractor3": [d[2] for d in distractors_list],
            "similarity_score1": similarity_scores1,
            "similarity_score2": similarity_scores2,
            "similarity_score3": similarity_scores3,
            "good_answer_prob": good_answer_score,
            "probability_score1": probability_score1,
            "probability_score2": probability_score2,
            "probability_score3": probability_score3,
            "lexical_similarity_score1": lexical_similarity_scores1,
            "lexical_similarity_score2": lexical_similarity_scores2,
            "lexical_similarity_score3": lexical_similarity_scores3,
        }
    )

    # Compute summary statistics for analysis
    summary_stats = {
        "mean_similarity": df_results[
            ["similarity_score1", "similarity_score2", "similarity_score3"]
        ]
        .mean()
        .mean(),  # type:ignore
        "min_similarity": df_results[
            ["similarity_score1", "similarity_score2", "similarity_score3"]
        ]
        .min()
        .min(),
        "max_similarity": df_results[
            ["similarity_score1", "similarity_score2", "similarity_score3"]
        ]
        .max()
        .max(),
        "mean_good_answer_prob": df_results[["good_answer_prob"]].mean().mean(),  # type:ignore
        "mean_probabilities": df_results[
            ["probability_score1", "probability_score2", "probability_score3"]
        ]
        .mean()
        .mean(),  # type:ignore
        "min_probabilities": df_results[
            ["probability_score1", "probability_score2", "probability_score3"]
        ]
        .min()
        .min(),
        "max_probabilities": df_results[
            ["probability_score1", "probability_score2", "probability_score3"]
        ]
        .max()
        .max(),
        "mean_lexical_similarity": df_results[
            [
                "lexical_similarity_score1",
                "lexical_similarity_score2",
                "lexical_similarity_score3",
            ]
        ]
        .mean()
        .mean(),  # type:ignore
        "min_lexical_similarity": df_results[
            [
                "lexical_similarity_score1",
                "lexical_similarity_score2",
                "lexical_similarity_score3",
            ]
        ]
        .min()
        .min(),
        "max_lexical_similarity": df_results[
            [
                "lexical_similarity_score1",
                "lexical_similarity_score2",
                "lexical_similarity_score3",
            ]
        ]
        .max()
        .max(),
    }

    return df_results, summary_stats


def detect_outliers_zscore(df_results, threshold=3):
    """
    Detect outliers using Z-score thresholding.

    Args:
        df_results (DataFrame): Data with scores.
        threshold (float): Z-score threshold to define outliers.

    Returns:
        DataFrame: Subset of outlier rows.
    """
    score_columns = [
        "similarity_score1",
        "similarity_score2",
        "similarity_score3",
        "probability_score1",
        "probability_score2",
        "probability_score3",
        "lexical_similarity_score1",
        "lexical_similarity_score2",
        "lexical_similarity_score3",
    ]
    all_scores = df_results[score_columns]
    z_scores = np.abs((all_scores - all_scores.mean()) / all_scores.std())
    outliers = df_results[(z_scores > threshold).any(axis=1)]
    return outliers


def save_plot(figure, filename):
    """
    Save a matplotlib figure to a file and close the figure to release memory.
    """
    figure.tight_layout()
    figure.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")


def plot_histogram(data, bins, kde, title, xlabel, ylabel, filename):
    """
    Generate and save a histogram.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(data, bins=bins, kde=kde)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_plot(plt, filename)


def visualize_results(df_results, outliers, config: Config, model_name=""):
    """
    Create and save visualizations, write outliers to a file
    and write twenty random samples to a file for manual analysis.

    Args:
        df_results (pd.DataFrame): DataFrame with similarity, probability, and lexical scores.
        outliers (pd.DataFrame): DataFrame with identified outlier rows from df_results.
        model_name (str): Optional prefix like Bert to add to saved filenames.

    Returns:
        None
    """

    # Add prefix to all file names
    prefix = f"{model_name}_" if model_name else ""
    dir_path = os.path.join(config.evaluation_directory, model_name) + "/"
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Calculate mean scores
    df_results["mean_similarity"] = df_results[
        ["similarity_score1", "similarity_score2", "similarity_score3"]
    ].mean(axis=1)

    df_results["mean_probabilities"] = df_results[
        ["probability_score1", "probability_score2", "probability_score3"]
    ].mean(axis=1)

    df_results["mean_lexical_similarity"] = df_results[
        [
            "lexical_similarity_score1",
            "lexical_similarity_score2",
            "lexical_similarity_score3",
        ]
    ].mean(axis=1)

    # Get data into proper format for the boxplot
    similarity_scores_combined = pd.concat(
        [
            df_results["similarity_score1"],
            df_results["similarity_score2"],
            df_results["similarity_score3"],
        ],
        ignore_index=True,
    )

    probability_scores_combined = pd.concat(
        [
            df_results["probability_score1"],
            df_results["probability_score2"],
            df_results["probability_score3"],
        ],
        ignore_index=True,
    )

    lexical_similarity_scores_combined = pd.concat(
        [
            df_results["lexical_similarity_score1"],
            df_results["lexical_similarity_score2"],
            df_results["lexical_similarity_score3"],
        ],
        ignore_index=True,
    )

    boxplot_data = pd.DataFrame(
        {
            "Similarity Scores": similarity_scores_combined,
            "Probabilities Scores": probability_scores_combined,
            "Lexical Similarity Scores": lexical_similarity_scores_combined,
        }
    )

    # Plot 1: Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=boxplot_data)
    plt.title(
        f"{prefix} Boxplot of Semantic Similarity, Probabilities, and Lexical Similarity Scores"
    )
    plt.ylabel("Score")
    save_plot(plt, f"{dir_path}boxplot_similarity_Probabilities_lexical_scores.png")

    # Plot 2-4: Histograms
    plot_histogram(
        similarity_scores_combined,
        bins=20,
        kde=True,
        title=f"{prefix} Histogram of Semantic Similarity Scores",
        xlabel="Semantic Similarity Score",
        ylabel="Frequency",
        filename=f"{dir_path}histogram_similarity_scores.png",
    )

    plot_histogram(
        probability_scores_combined,
        bins=20,
        kde=True,
        title=f"{prefix} Histogram of Probabilities Scores",
        xlabel="Probabilities Score",
        ylabel="Frequency",
        filename=f"{dir_path}histogram_Probability_scores.png",
    )

    plot_histogram(
        lexical_similarity_scores_combined,
        bins=20,
        kde=True,
        title=f"{prefix} Histogram of Lexical Similarity Scores",
        xlabel="Lexical Similarity Score",
        ylabel="Frequency",
        filename=f"{dir_path}histogram_lexical_similarity_scores.png",
    )

    # Plot 5: Correlation Heatmap
    correlation_matrix = df_results[
        ["mean_similarity", "mean_probabilities", "mean_lexical_similarity"]
    ].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title(
        f"{prefix} Correlation Heatmap of Average Scores (Similarity, Probabilities, Lexical Similarity)"
    )
    save_plot(plt, f"{dir_path}correlation_heatmap_avg_scores.png")

    # Save Outliers to a file
    if not outliers.empty:
        with open(f"{dir_path}outliers.txt", "w", encoding="utf-8") as f:
            f.write("Outliers Detected:\n")
            total_outliers = 0
            for _, row in outliers.iterrows():
                f.write(f"Context: {row['support']}\n")
                f.write(f"Question: {row['question']}\n")
                f.write(f"Correct Answer: {row['correct_answer']}\n")
                f.write(
                    f"Distractors: {row['distractor1']}, {row['distractor2']}, {row['distractor3']}\n"
                )

                f.write(
                    f"  - {row['distractor1']}: Similarity Score: {row['similarity_score1']:.6f}\n"
                )
                f.write(
                    f"  - {row['distractor2']}: Similarity Score: {row['similarity_score2']:.6f}\n"
                )
                f.write(
                    f"  - {row['distractor3']}: Similarity Score: {row['similarity_score3']:.6f}\n\n"
                )

                f.write(
                    f"  - {row['correct_answer']}: Probability Score (good answer): {row['good_answer_prob']:.6f}\n"
                )
                f.write(
                    f"  - {row['distractor1']}: Probability Score: {row['probability_score1']:.6f}\n"
                )
                f.write(
                    f"  - {row['distractor2']}: Probability Score: {row['probability_score2']:.6f}\n"
                )
                f.write(
                    f"  - {row['distractor3']}: Probability Score: {row['probability_score3']:.6f}\n\n"
                )

                f.write(
                    f"  - {row['distractor1']}: Lexical Score: {row['lexical_similarity_score1']:.6f}\n"
                )
                f.write(
                    f"  - {row['distractor2']}: Lexical Score: {row['lexical_similarity_score2']:.6f}\n"
                )
                f.write(
                    f"  - {row['distractor3']}: Lexical Score: {row['lexical_similarity_score3']:.6f}\n\n"
                )

                total_outliers += 1

            f.write(f"Total count of outliers: {total_outliers}\n")
        print(f"Saved: {dir_path}outliers.txt")
    else:
        print("No outliers detected.")

    # Save 20 random samples to a file
    random_samples = random.sample(df_results.to_dict("records"), 20)
    with open(f"{dir_path}random_samples.txt", "w", encoding="utf-8") as f:
        f.write("Random Samples:\n")
        for sample in random_samples:
            f.write(f"Context: {sample['support']}\n")
            f.write(f"Question: {sample['question']}\n")
            f.write(f"Correct Answer: {sample['correct_answer']}\n")
            f.write(
                f"Distractors: {sample['distractor1']}, {sample['distractor2']}, {sample['distractor3']}\n"
            )
            f.write(
                f"  - {sample['distractor1']}: Similarity Score: {sample['similarity_score1']:.6f}\n"
            )
            f.write(
                f"  - {sample['distractor2']}: Similarity Score: {sample['similarity_score2']:.6f}\n"
            )
            f.write(
                f"  - {sample['distractor3']}: Similarity Score: {sample['similarity_score3']:.6f}\n\n"
            )
            f.write(
                f"  - {sample['correct_answer']}: Probability Score (good answer): {sample['good_answer_prob']:.6f}\n"
            )
            f.write(
                f"  - {sample['distractor1']}: Probability Score: {sample['probability_score1']:.6f}\n"
            )
            f.write(
                f"  - {sample['distractor2']}: Probability Score: {sample['probability_score2']:.6f}\n"
            )
            f.write(
                f"  - {sample['distractor3']}: Probability Score: {sample['probability_score3']:.6f}\n\n"
            )
            f.write(
                f"  - {sample['distractor1']}: Lexical Score: {sample['lexical_similarity_score1']:.6f}\n"
            )
            f.write(
                f"  - {sample['distractor2']}: Lexical Score: {sample['lexical_similarity_score2']:.6f}\n"
            )
            f.write(
                f"  - {sample['distractor3']}: Lexical Score: {sample['lexical_similarity_score3']:.6f}\n\n"
            )
        print(f"Saved: {dir_path}random_samples.txt")


def print_statistics(summary_stats):
    """
    Print formatted summary statistics.
    """
    print("=== Combined Similarity, Probability and Lexical Statistics ===")

    stats = [
        ("Mean Similarity Score", summary_stats["mean_similarity"]),
        ("Min Similarity Score", summary_stats["min_similarity"]),
        ("Max Similarity Score", summary_stats["max_similarity"]),
        (
            "Mean Probability Score (Good Answer)",
            summary_stats["mean_good_answer_prob"],
        ),
        ("Mean Probability Score", summary_stats["mean_probabilities"]),
        ("Min Probability Score", summary_stats["min_probabilities"]),
        ("Max Probability Score", summary_stats["max_probabilities"]),
        ("Mean Lexical Similarity Score", summary_stats["mean_lexical_similarity"]),
        ("Min Lexical Similarity Score", summary_stats["min_lexical_similarity"]),
        ("Max Lexical Similarity Score", summary_stats["max_lexical_similarity"]),
    ]

    for stat, value in stats:
        print(f"{stat:<40}: {value:.4f}")


def evaluate(config: Config):
    # if len(sys.argv) < 3:
    #     print("Usage: python3 evaluation.py <dataset.arrow> <model_name>")
    #     sys.exit(1)

    # filename = sys.argv[1]
    # model_name = sys.argv[2]

    # dataset = load_arrow_dataset(filename)

    dataset = load_from_disk(utils.get_predictions_directory_name(config))
    model_name = config.model.value

    # Load models
    roberta_model = load_model()
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Process dataset
    df_results, summary_stats = process_dataset(dataset, roberta_model, qa_model)

    # Print statistics
    print_statistics(summary_stats)

    # Detect outliers
    outliers = detect_outliers_zscore(df_results)

    # Save visualizations and outlier report with model name prefix
    visualize_results(df_results, outliers, config, model_name=model_name)
