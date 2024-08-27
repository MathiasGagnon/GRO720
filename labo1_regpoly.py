import numpy as np
from typing import List
from matplotlib import pyplot as plt


def loss_function(y_pred: float, y_exp: float) -> float:
    return np.sum(((y_pred - y_exp) ** 2))


def gradient_function(y_pred: float, y_exp: float, X: np.matrix) -> float:
    return 2 * np.sum((y_pred - y_exp) * X)


def regression_polynomial(N: int, data, nbr_iteration: int, lr: float):
    A = np.matrix([0.0] * (N + 1))
    losses = []

    for _ in range(nbr_iteration):
        total_loss = 0
        total_gradient = 0
        for x, y in data:
            X = [x] * (N + 1)
            for i in range(N, -1, -1):
                X[i] = X[i] ** i

            X = np.matrix(X)
            y_pred = np.matmul(A, X.T)

            total_gradient += gradient_function(y_pred, y, X)
            total_loss += loss_function(y_pred, y)

        A -= lr * total_gradient
        losses.append(total_loss)

    return A, losses


def test_polynomial_degrees(
    data, degrees: List[int], learning_rate: float, num_iterations: int
):
    x_range = np.linspace(-1.25, 1.25, 100)

    # Plot polynomial regression lines
    plt.figure(figsize=(12, 8))
    plt.scatter(data[:, 0], data[:, 1], color="red", label="Training data")

    for N in degrees:
        # Train the model for polynomial degree N
        A, losses = regression_polynomial(N, data, num_iterations, learning_rate)

        # Generate predictions
        y_predicted = []
        for x in x_range:
            X = np.array([x**i for i in range(N + 1)])  # Degree N polynomial
            X = np.matrix(X)
            y_pred = np.matmul(A, X.T)
            y_predicted.append(y_pred[0, 0])

        # Plot the polynomial regression line
        plt.plot(x_range, y_predicted, label=f"Polynomial Degree {N}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial Regression pour different N")
    plt.legend()
    plt.show()

    # Plot loss curves
    plt.figure(figsize=(12, 8))
    for N in degrees:
        _, losses = regression_polynomial(N, data, num_iterations, learning_rate)
        plt.plot(range(num_iterations), losses, label=f"Degree {N}")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss pour different N")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    data = np.array(
        [
            [-0.95, +0.02],
            [-0.82, +0.03],
            [-0.62, -0.17],
            [-0.43, -0.12],
            [-0.17, -0.37],
            [-0.07, -0.25],
            [+0.25, -0.10],
            [+0.38, +0.14],
            [+0.61, +0.53],
            [+0.79, +0.71],
            [+1.04, +1.53],
        ]
    )
    
    learning_rate = 0.001
    num_iterations = 100
    degrees = [1, 2, 7, 10, 11]

    test_polynomial_degrees(data, degrees, learning_rate, num_iterations)
