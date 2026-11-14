
import pandas as pd

from ml.gradient_descent_engine import train_gradient_descent

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_models(df, target_column):

    # Separate features and target
    if target_column not in df.columns:
        raise ValueError("The selected target column was not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Remove rows where the target is missing
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if len(X) < 2:
        raise ValueError("At least two rows with a valid target are required.")

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression training requires a numerical target.")

    if X.shape[1] == 0:
        raise ValueError("No input features are available.")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Identify feature types
    numerical_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    if not numerical_features and not categorical_features:
        raise ValueError(
            "No supported numerical or categorical input features were found."
        )

    # Build preprocessing pipeline
    transformers = []

    if numerical_features:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        transformers.append(
            ("numerical", numerical_pipeline, numerical_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    # Candidate models
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    results = []
    fitted_models = {}

    # Train and evaluate scikit-learn models
    for model_name, model in models.items():

        preprocessor = ColumnTransformer(transformers)

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        })

        fitted_models[model_name] = pipeline

    # Custom Gradient Descent Linear Regression
    predictions, loss_history = train_gradient_descent(
        X_train,
        X_test,
        y_train,
        y_test
    )

    gd_mae = mean_absolute_error(y_test, predictions)
    gd_rmse = mean_squared_error(y_test, predictions) ** 0.5
    gd_r2 = r2_score(y_test, predictions)

    results.append({
        "Model": "Gradient Descent Linear Regression",
        "MAE": gd_mae,
        "RMSE": gd_rmse,
        "R²": gd_r2,
        "Initial Loss": loss_history[0],
        "Final Loss": loss_history[-1],
        "Iterations": len(loss_history) - 1
    })

    # Create comparison table
    results_df = pd.DataFrame(results)

    # Select the lowest-MAE scikit-learn model
    best_model_name = min(
        fitted_models,
        key=lambda name: results_df.loc[
            results_df["Model"] == name, "MAE"
        ].iloc[0]
    )

    best_model = fitted_models[best_model_name]

    # Refit the selected model on all rows with valid targets.
    # Its displayed test metrics remain based on the held-out test set.
    best_model.fit(X, y)

    return results_df, loss_history, best_model, best_model_name
