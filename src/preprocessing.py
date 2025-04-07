from datasets import DatasetDict
from utils import dataset_to_df, df_to_dataset
import polars as pl


def preprocess_encoder_decoder(dataset: DatasetDict):
    """
    Preprocess encoder-decoder model

    The dataset will be converted so there wont be three distractors per row, but a
    single distractor. To keep all distractors, the original row will be split into
    three rows. For the encoder decoder model we believe this to be beneficial, so it
    can focus on generating small sequences of tokens that don't vary too much in length.
    Those lengths may vary more when predicting multiple answer options at once.

    So if the current dataset looks like this: `A B C1 C2 C3`, it will become like this:

    ```
    A B C1
    A B C2
    A B C3
    ```
    """
    df = dataset_to_df(dataset)

    df = df.unpivot(
        index=["question", "correct_answer", "support", "split"],
        on=["distractor1", "distractor2", "distractor3"],
        variable_name="x",
        value_name="distractor",
    )

    df = df.drop(["x"])

    return df_to_dataset(df)


def preprocess_decoder(dataset: DatasetDict) -> DatasetDict:
    """
    Preprocess decoder model
    """
    df = dataset_to_df(dataset)

    # We are creating two new columns: the text column with the input for the model such as the
    # the support, question and answer. The target for the model will be the three distractors.
    # With these transformations we can easily tokenize the data inside the tokenize function.

    def text(is_baseline):
        text = (
            "There will be a question, support and correct answer. Given those generate 3 false answer options in the following format:\n\nDistractor 1: <answer>\nDistractor 2: <answer>\nDistractor 3: <answer>\n\n"
            if is_baseline
            else ""
        )

        text += "Question: {}\nCorrect Answer: {}\nSupport: {}\n"

        text += "" if is_baseline else "Generate 3 false answer options:\n"

        return text

    df = df.with_columns(
        [
            pl.format(
                text(False),
                pl.col("question"),
                pl.col("correct_answer"),
                pl.col("support"),
            ).alias("text"),
            pl.format(
                text(True),
                pl.col("question"),
                pl.col("correct_answer"),
                pl.col("support"),
            ).alias("baseline"),
            pl.format(
                "Distractor 1: {}\nDistractor 2: {}\nDistractor 3: {}",
                pl.col("distractor1"),
                pl.col("distractor2"),
                pl.col("distractor3"),
            ).alias("target"),
        ]
    )

    # the following columns are no longer needed since they are used in the creation of the new columns
    df = df.drop(
        [
            "distractor1",
            "distractor2",
            "distractor3",
            "question",
            "correct_answer",
            "support",
        ]
    )

    return df_to_dataset(df)
