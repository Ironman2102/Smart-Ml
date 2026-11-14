
import pandas as pd


def determine_problem_type(df, target_column):
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    target = df[target_column].dropna()

    if target.empty:
        raise ValueError(
            "The target column contains no valid values."
        )

    # Numeric targets are regression by default.
    # This avoids misclassifying small continuous datasets.
    if pd.api.types.is_numeric_dtype(target):
        return "Regression"

    # Non-numeric targets are treated as classification.
    return "Classification"
