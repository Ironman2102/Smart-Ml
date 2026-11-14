
import pandas as pd


def analyze_dataset(df, target_column=None):
    analysis = {}

    # Basic dataset information
    analysis["rows"] = df.shape[0]
    analysis["columns"] = df.shape[1]
    analysis["missing_values"] = df.isnull().sum()

    if len(df) > 0:
        analysis["missing_percentages"] = (
            df.isnull().mean().mul(100).round(2)
        )
    else:
        analysis["missing_percentages"] = pd.Series(
            0.0, index=df.columns
        )

    analysis["data_types"] = df.dtypes

    analysis["numerical_columns"] = (
        df.select_dtypes(include="number").columns.tolist()
    )

    analysis["categorical_columns"] = (
        df.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()
    )

    analysis["duplicate_rows"] = int(df.duplicated().sum())
    analysis["unique_values"] = df.nunique()

    # Numerical statistics
    numerical_columns = analysis["numerical_columns"]

    if numerical_columns:
        analysis["statistics"] = df[numerical_columns].describe()
    else:
        analysis["statistics"] = pd.DataFrame()

    # Outlier detection using the IQR method
    outliers = {}
    outlier_details = {}

    for column in numerical_columns:
        values = df[column].dropna()

        if values.empty:
            outliers[column] = 0
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        ).fillna(False)

        flagged_values = df.loc[mask, column]
        outliers[column] = int(mask.sum())

        if not flagged_values.empty:
            outlier_details[column] = {
                "count": int(mask.sum()),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "examples": [
                    {
                        "row_index": str(index),
                        "value": (
                            value.item()
                            if hasattr(value, "item")
                            else value
                        )
                    }
                    for index, value
                    in flagged_values.head(10).items()
                ]
            }

    analysis["outliers"] = pd.Series(outliers)
    analysis["outlier_details"] = outlier_details

    # Correlation analysis
    if len(numerical_columns) > 1:
        analysis["correlation"] = df[numerical_columns].corr()
    else:
        analysis["correlation"] = pd.DataFrame()

    # Potential feature columns
    analysis["potential_features"] = [
        column
        for column in df.columns
        if column != target_column
    ]

    # Numeric target correlations
    if (
        target_column is not None
        and target_column in df.columns
        and pd.api.types.is_numeric_dtype(df[target_column])
        and target_column in numerical_columns
    ):
        target_correlations = (
            df[numerical_columns]
            .corr()[target_column]
            .drop(target_column, errors="ignore")
            .sort_values(key=abs, ascending=False)
        )
        analysis["target_correlations"] = target_correlations
    else:
        analysis["target_correlations"] = pd.Series(dtype=float)

    return analysis
