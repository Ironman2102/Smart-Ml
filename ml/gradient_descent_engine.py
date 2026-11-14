
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.optimizers import gradient_descent, predict


def train_gradient_descent(X_train, X_test, y_train, y_test):
    """Preprocess data and train a linear regression model using Gradient Descent."""

    # Identify numerical and categorical columns.
    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    # Numerical preprocessing.
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing steps.
    transformers = []

    if numerical_features:
        transformers.append(
            ("numerical", numerical_pipeline, numerical_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError("No usable input features were found.")

    preprocessor = ColumnTransformer(transformers)

    preparation_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("scaler", StandardScaler(with_mean=False))
    ])

    # Fit preprocessing ONLY on training data.
    X_train_processed = preparation_pipeline.fit_transform(X_train)

    # Apply the same transformations to test data.
    X_test_processed = preparation_pipeline.transform(X_test)

    # Convert sparse matrices to NumPy arrays.
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Convert target values to numeric arrays.
    y_train_array = np.asarray(y_train, dtype=float).reshape(-1)

    # Train using the custom Gradient Descent optimizer.
    weights, bias, loss_history = gradient_descent(
        X_train_processed,
        y_train_array,
        learning_rate=0.01,
        iterations=5000,
        tolerance=1e-8
    )

    # Generate predictions on unseen test data.
    predictions = predict(
        X_test_processed,
        weights,
        bias
    )

    return predictions, loss_history
