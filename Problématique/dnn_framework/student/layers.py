import numpy as np

from dnn_framework.layer import Layer

epsilon = 1e-6


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.param = {
            "w": np.random.randn(output_count, input_count)
            * np.sqrt(2 / (input_count + output_count)),
            "b": np.random.randn(output_count) * np.sqrt(2 / output_count),
        }

    def get_parameters(self):
        return self.param

    def get_buffers(self):
        return {}

    def forward(self, x):
        return np.matmul(x, self.param["w"].T) + self.param["b"], x

    def backward(self, output_grad, cache):
        return np.dot(output_grad, self.param["w"]), {
            "w": np.dot(output_grad.T, cache),
            "b": np.sum(output_grad, axis=0),
        }


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.epsilon = 1 * 10**-6
        self.alpha = alpha

        self.parameters = {"gamma": np.ones(input_count), "beta": np.zeros(input_count)}
        self.buffers = {
            "global_mean": np.zeros(input_count),
            "global_variance": np.zeros(input_count),
        }

        self.first = True

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return self.buffers

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        return self._forward_evaluation(x)

    def _forward_training(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        if self.first:
            self.buffers["global_mean"] = batch_mean
            self.buffers["global_variance"] = batch_var
            self.first = False
        else:
            self.buffers["global_mean"] = (
                (1 - self.alpha) * self.buffers["global_mean"] + self.alpha * batch_mean
            )
            self.buffers["global_variance"] = (
                (1 - self.alpha) * self.buffers["global_variance"] + self.alpha * batch_var
            )

        x_hat = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
        y = self.parameters["gamma"] * x_hat + self.parameters["beta"]
        return y, (x, x_hat)

    def _forward_evaluation(self, x):
        x_hat = (x - self.buffers["global_mean"]) / np.sqrt(
            self.buffers["global_variance"] + self.epsilon
        )
        return self.parameters["gamma"] * x_hat + self.parameters["beta"], (
            x,
            x_hat
        )

    def backward(self, output_grad, cache):
        x, x_hat = cache
        M = x.shape[0]

        grad_x_hat = output_grad * self.parameters["gamma"]
        grad_var = np.sum(
            grad_x_hat
            * (x - self.buffers["global_mean"])
            * (-1 / 2)
            * (self.buffers["global_variance"] + self.epsilon) ** (-3 / 2),
            axis=0,
        )
        grad_mean = -np.sum(
            grad_x_hat / np.sqrt(self.buffers["global_variance"] + self.epsilon),
            axis=0,
        )
        grad_x = (
            grad_x_hat / np.sqrt(self.buffers["global_variance"] + self.epsilon)
            + 2 / M * grad_var * (x - self.buffers["global_mean"])
            + 1 / M * grad_mean
        )

        return grad_x, {"gamma": np.sum(output_grad * x_hat, axis=0), "beta":  np.sum(output_grad, axis=0)}


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        return 1 / (1 + np.exp(-x)), x

    def backward(self, output_grad, cache):
        return self.forward(cache)[0] * (1 - self.forward(cache)[0]) * output_grad, {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        return np.maximum(0, x), x

    def backward(self, output_grad, cache):
        return np.where(cache > 0, 1, 0) * output_grad, {}
