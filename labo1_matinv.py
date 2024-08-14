import numpy as np
from matplotlib import pyplot as plt

def loss_function(mat_a, mat_b, mat_i):
    return np.sum(np.square(np.matmul(mat_a, mat_b) - mat_i))

def gradient(mat_a, mat_b, mat_i):
    return 2 * np.matmul((np.matmul(mat_a, mat_b) - mat_i), mat_a.T)

def optimisation_step(mat_a, mat_b, mat_i, mu):
    updated_b = mat_b - mu * gradient(mat_a, mat_b, mat_i)
    loss = loss_function(mat_a, updated_b, mat_i)
    return updated_b, loss

def optimise(mat_a, mat_b, mat_i, mu, num_iterations):
    iteration_b = mat_b
    losses = []
    iterations_b = []
    x_axis = list(range(num_iterations))
    min_loss = float('inf')

    for _ in range(num_iterations):
        iteration_b, loss = optimisation_step(mat_a, iteration_b, mat_i, mu)
        losses.append(loss)
        if loss < min_loss:
            min_loss = loss
        
        iterations_b.append(iteration_b)

    plt.plot(x_axis, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Optimization with mu={mu}')
    plt.show()

    return min_loss

def run_optimization(matrix, mu_values, num_iterations):
    identity_matrix = np.identity(matrix.shape[0])
    expected_b = matrix.I
    print('Expected b matrix, as calculated by numpy')
    print(expected_b)
    
    expected_loss = loss_function(matrix, expected_b, identity_matrix)
    print('\nExpected b matrix loss')
    print(expected_loss)
    
    for mu in mu_values:
        print(f'\nOptimisation of {matrix.shape[0]}x{matrix.shape[1]} matrix with {num_iterations} iterations and mu of {mu}')
        b_matrix_optimised = np.random.rand(matrix.shape[0], matrix.shape[1])
        min_loss = optimise(matrix, b_matrix_optimised, identity_matrix, mu, num_iterations)
        print(f'Smallest loss: {min_loss}')

matrices = {
    "3x3": np.matrix([[3,4,1],[5,2,3],[6,2,2]]),
    "6x6": np.matrix([[3,4,1, 2, 1, 5],[5,2,3, 2, 2, 1],[6,2,2, 6, 4, 5], [1, 2, 1, 3, 1, 2], [1, 5, 2, 3, 3, 3], [1, 2, 2, 4, 2, 1]]),
    "4x4": np.matrix([[2, 1, 1, 2],[1, 2, 3, 2],[2, 1, 1, 2], [3, 1, 4, 1]])
}

mu_values = [0.005, 0.001, 0.01]
num_iterations = 1000

for size, matrix in matrices.items():
    run_optimization(matrix, mu_values, num_iterations)
