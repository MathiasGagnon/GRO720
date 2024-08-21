import numpy as np
from matplotlib import pyplot as plt


def loss_function(y_pred, y_exp):
    return np.sum(((y_pred - y_exp) ** 2))


def gradient_function(y_pred, y_exp, x, x_exponent):
    return 2 * np.sum((y_pred - y_exp) * x**x_exponent)


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

x = data[:, 0]
y_exp = data[:, 1]

theta_0 = 0.0
theta_1 = 0.0
theta_2 = 0.0
theta_3 = 0.0
theta_4 = 0.0
theta_5 = 0.0
theta_6 = 0.0
theta_7 = 0.0
learning_rate = 0.01
num_iterations = 10000
min_loss = float("inf")

for i in range(num_iterations):
    y_pred = theta_0 + theta_1 * x
    loss = loss_function(y_pred, y_exp)
    grad_theta_0 = gradient_function(y_pred, y_exp, x, 0)
    grad_theta_1 = gradient_function(y_pred, y_exp, x, 1)

    theta_0 -= learning_rate * grad_theta_0
    theta_1 -= learning_rate * grad_theta_1
    if loss < min_loss:
        min_loss = loss

print(f"Min loss: {min_loss}")
plt.scatter(x, y_exp, color="blue", label="Data")
plt.plot(x, theta_0 + theta_1 * x, color="red", label="Prediction model")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Prediction avec un polynôme de degré 1")
plt.show()


for i in range(num_iterations):
    y_pred = theta_0 + theta_1 * x
    loss = loss_function(y_pred, y_exp)
    grad_theta_0 = gradient_function(y_pred, y_exp, x, 0)
    grad_theta_1 = gradient_function(y_pred, y_exp, x, 1)
    grad_theta_2 = gradient_function(y_pred, y_exp, x, 2)

    theta_0 -= learning_rate * grad_theta_0
    theta_1 -= learning_rate * grad_theta_1
    theta_2 -= learning_rate * grad_theta_2
    if loss < min_loss:
        min_loss = loss

print(f"Min loss: {min_loss}")
plt.scatter(x, y_exp, color="blue", label="Data")
plt.plot(
    x, theta_0 + theta_1 * x + theta_2 * x**2, color="red", label="Prediction model"
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Prediction avec un polynôme de degré 7")
plt.show()

for i in range(num_iterations):
    y_pred = theta_0 + theta_1 * x
    loss = loss_function(y_pred, y_exp)
    grad_theta_0 = gradient_function(y_pred, y_exp, x, 0)
    grad_theta_1 = gradient_function(y_pred, y_exp, x, 1)
    grad_theta_2 = gradient_function(y_pred, y_exp, x, 2)
    grad_theta_3 = gradient_function(y_pred, y_exp, x, 3)
    grad_theta_4 = gradient_function(y_pred, y_exp, x, 4)
    grad_theta_5 = gradient_function(y_pred, y_exp, x, 5)
    grad_theta_6 = gradient_function(y_pred, y_exp, x, 6)
    grad_theta_7 = gradient_function(y_pred, y_exp, x, 7)

    theta_0 -= learning_rate * grad_theta_0
    theta_1 -= learning_rate * grad_theta_1
    theta_2 -= learning_rate * grad_theta_2
    theta_3 -= learning_rate * grad_theta_3
    theta_4 -= learning_rate * grad_theta_4
    theta_5 -= learning_rate * grad_theta_5
    theta_6 -= learning_rate * grad_theta_6
    theta_7 -= learning_rate * grad_theta_7

    if loss < min_loss:
        min_loss = loss

print(f"Min loss: {min_loss}")
plt.scatter(x, y_exp, color="blue", label="Data")
plt.plot(
    x,
    theta_0
    + theta_1 * x
    + theta_2 * x**2
    + theta_3 * x**3
    + theta_4 * x**4
    + theta_5 * x**5
    + theta_6 * x**6
    + theta_7 * x**7,
    color="red",
    label="Prediction model",
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Prediction avec un polynôme de degré 2")
plt.show()
