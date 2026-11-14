import pandas as pd


def analyze_features(df, target_column):

    feature_analysis = []

    for column in df.columns:

        # Skip target
        if column == target_column:
            continue

        series = df[column]

        # -------------------------
        # Basic information
        # -------------------------

        data_type = str(series.dtype)
        unique_count = series.nunique()
        missing_count = series.isnull().sum()

        # -------------------------
        # Determine feature type
        # -------------------------

        if pd.api.types.is_numeric_dtype(series):

            feature_type = "Numerical"

        else:

            feature_type = "Categorical/Text"

        # -------------------------
        # Possible ID detection
        # -------------------------

        possible_id = False

        if unique_count == len(df):

            possible_id = True

        feature_analysis.append({
            "Feature": column,
            "Type": feature_type,
            "Unique Values": unique_count,
            "Missing Values": missing_count,
            "Possible ID": possible_id
        })

    return pd.DataFrame(feature_analysis)