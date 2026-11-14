
import numpy as np


def predict(X, weights, bias):
    """
    Generate predictions using:
    y = Xw + b
    """
    return np.dot(X, weights) + bias


def mean_squared_error(X, y, weights, bias):
    """
    Calculate Mean Squared Error (MSE).
    """
    predictions = predict(X, weights, bias)

    errors = predictions - y

    mse = np.mean(errors ** 2)

    return mse


def compute_gradients(X, y, weights, bias):
    """
    Calculate partial derivatives of MSE
    with respect to weights and bias.
    """
    m = len(y)

    predictions = predict(X, weights, bias)

    errors = predictions - y

    # Partial derivatives for weights
    dw = (2 / m) * np.dot(X.T, errors)

    # Partial derivative for bias
    db = (2 / m) * np.sum(errors)

    return dw, db



def gradient_descent(
    X,
    y,
    learning_rate=0.01,
    iterations=5000,
    tolerance=1e-8
):
    """Train linear regression using numerically stable batch gradient descent."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if len(X) != len(y) or len(y) == 0:
        raise ValueError("X and y must have matching non-empty rows.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")

    if (not np.isfinite(learning_rate) or learning_rate <= 0
            or iterations <= 0 or tolerance < 0):
        raise ValueError("Invalid optimization settings.")

    m = len(y)

    weights = np.zeros(X.shape[1])
    bias = 0.0

    # Include the bias term when estimating a safe learning rate.
    X_augmented = np.column_stack((X, np.ones(m)))

    largest_eigenvalue = (
        2 * np.linalg.norm(X_augmented, ord=2) ** 2 / m
    )

    if not np.isfinite(largest_eigenvalue) or largest_eigenvalue <= 0:
        raise ValueError("Unable to calculate a stable learning rate.")

    safe_learning_rate = 0.9 / largest_eigenvalue

    # Never use a learning rate larger than the estimated safe value.
    actual_learning_rate = min(learning_rate, safe_learning_rate)

    loss_history = []

    previous_loss = mean_squared_error(X, y, weights, bias)
    loss_history.append(previous_loss)

    for i in range(iterations):

        dw, db = compute_gradients(X, y, weights, bias)

        weights -= actual_learning_rate * dw
        bias -= actual_learning_rate * db

        current_loss = mean_squared_error(X, y, weights, bias)

        if not np.isfinite(current_loss):
            raise FloatingPointError(
                "Training diverged. Check your input data and scaling."
            )

        loss_history.append(current_loss)

        if abs(previous_loss - current_loss) < tolerance:
            break

        previous_loss = current_loss

    return weights, bias, loss_history
