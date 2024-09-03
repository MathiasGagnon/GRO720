import numpy as np
from typing import List
from matplotlib import pyplot as plt


def loss_function(mat_a: np.matrix, mat_b: np.matrix):
    return np.sum(np.square(np.matmul(mat_b, mat_a) - np.identity(mat_a.shape[0])))


def gradient(mat_a: np.matrix, mat_b: np.matrix):
    return 2 * np.matmul(
        (np.matmul(mat_b, mat_a) - np.identity(mat_a.shape[0])), mat_a.T
    )


def optimise(mat_a: np.matrix, mat_b: np.matrix, mu: float, num_iterations: int = 1000):
    losses = []

    for _ in range(num_iterations):
        # Vérification des valeurs NaN et Overflows
        if np.any(np.isnan(mat_b)) or np.any(np.isinf(mat_b)):
            print("NaN ou overflow détecté. Arrêt de l'optimisation.")
            break

        mat_b = mat_b - mu * gradient(mat_a, mat_b)
        losses.append(loss_function(mat_a, mat_b))

    return mat_b, losses


def run_optimization(
    matrix: np.matrix, mu_values: List[float], num_iterations: int = 1000
):
    try:
        expected_b = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        print("No inverse matrix")
        return

    print("Expected b matrix, as calculated by numpy")
    print(expected_b)
    plt.figure()

    best_mu = None
    best_final_loss = float("inf")
    best_losses = None

    for mu in mu_values:
        print(
            f"\nOptimisation of {matrix.shape[0]}x{matrix.shape[1]} matrix with {num_iterations} iterations and mu of {mu}"
        )
        mat_b = (
            np.random.rand(matrix.shape[0], matrix.shape[1]) * 0.01
        )  # Initialisation proche de zéro
        _, losses = optimise(matrix, mat_b, mu, num_iterations)
        final_loss = losses[-1]

        # Track the best mu and corresponding loss
        if final_loss < best_final_loss:
            best_mu = mu
            best_final_loss = final_loss
            best_losses = losses

        plt.plot(range(num_iterations), losses, label=f"mu={mu}")

    # Plot the losses for the best mu
    plt.plot(
        range(num_iterations),
        best_losses,
        label=f"Best mu={best_mu}",
        linewidth=3,
        linestyle="--",
    )

    # Plot the expected loss as a horizontal line
    plt.axhline(y=0, color="r", linestyle="--", label="Expected Loss")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Optimization for {matrix.shape[0]}x{matrix.shape[1]} Matrix")
    plt.legend()
    plt.show()

    print(f"\nBest mu: {best_mu} with final loss: {best_final_loss}")


matrices = (
    np.matrix([[3, 4, 1], [5, 2, 3], [6, 2, 2]]),
    np.matrix(
        [
            [3, 4, 1, 2, 1, 5],
            [5, 2, 3, 2, 2, 1],
            [6, 2, 2, 6, 4, 5],
            [1, 2, 1, 3, 1, 2],
            [1, 5, 2, 3, 3, 3],
            [1, 2, 2, 4, 2, 1],
        ]
    ),
    np.matrix(
        [[2, 1, 1, 2], [1, 2, 3, 2], [2, 1, 1, 2], [3, 1, 4, 1]]
    ),  # N'a pas d'inverse, typiquement parce que le déterminant est zéro
)

mu_values = [0.0001, 0.0005, 0.001]
num_iterations = 1000

for matrix in matrices:
    run_optimization(matrix, mu_values, num_iterations)
