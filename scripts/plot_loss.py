
import numpy as np
import matplotlib.pyplot as plt

from ml.optimizers import gradient_descent

X = np.array([
    [1.0],
    [2.0],
    [3.0]
])

y = np.array([2.0, 4.0, 6.0])

learning_rates = [0.0001, 0.001, 0.01, 0.1]

for rate in learning_rates:

    weights, bias, loss_history = gradient_descent(
        X,
        y,
        learning_rate=rate,
        iterations=1000
    )

    plt.plot(
        range(len(loss_history)),
        loss_history,
        label=f"Learning Rate = {rate}"
    )

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent - Learning Rate Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
