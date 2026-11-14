import pandas as pd


def clean_dataset(df):

    cleaned_df = df.copy()

    cleaning_report = {}

    # -------------------------
    # Remove duplicate rows
    # -------------------------

    duplicate_count = cleaned_df.duplicated().sum()

    cleaned_df = cleaned_df.drop_duplicates()

    cleaning_report["duplicates_removed"] = duplicate_count

    # -------------------------
    # Handle missing values
    # -------------------------

    missing_before = cleaned_df.isnull().sum()

    for column in cleaned_df.columns:

        if cleaned_df[column].isnull().sum() == 0:
            continue

        # Numerical columns → median
        if pd.api.types.is_numeric_dtype(cleaned_df[column]):

            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

        # Categorical columns → most common value
        else:

            mode = cleaned_df[column].mode()

            if not mode.empty:
                cleaned_df[column] = cleaned_df[column].fillna(
                    mode[0]
                )

    cleaning_report["missing_values_before"] = missing_before
    cleaning_report["missing_values_after"] = (
        cleaned_df.isnull().sum()
    )

    return cleaned_df, cleaning_report