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
        self.epsilon = 1 * 10**-7
        self.alpha = alpha

        self.parameters = {"gamma": np.ones(input_count), "beta": np.zeros(input_count)}
        self.buffers = {
            "global_mean": np.zeros(input_count),
            "global_variance": np.zeros(input_count),
        }

        self.first = False

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

        self.buffers["global_mean"] = (
            (1 - self.alpha) * self.buffers["global_mean"] + self.alpha * batch_mean
            if self.first
            else batch_mean
        )
        self.buffers["global_variance"] = (
            (1 - self.alpha) * self.buffers["global_variance"] + self.alpha * batch_var
            if self.first
            else batch_var
        )

        self.first = True

        x_hat = (x - batch_mean) / (batch_var + self.epsilon) ** 0.5
        y = self.parameters["gamma"] * x_hat + self.parameters["beta"]
        return y, (x, x_hat)

    def _forward_evaluation(self, x):
        xhat = (x - self.buffers["global_mean"]) / (
            self.buffers["global_variance"] + self.epsilon
        ) ** 0.5
        return self.parameters["gamma"] * xhat + self.parameters["beta"], (
            x,
            xhat,
            self.buffers["global_mean"],
            self.buffers["global_variance"],
        )

    def backward(self, output_grad, cache):
        grad_dict = {}
        x, xhat = cache
        M = x.shape[0]

        grad_xhat = output_grad * self.parameters["gamma"]
        grad_var = np.sum(
            grad_xhat
            * (x - self.buffers["global_mean"])
            * (-1 / 2)
            * (self.buffers["global_variance"] + self.epsilon) ** (-3 / 2),
            axis=0,
        )
        grad_mean = -np.sum(
            grad_xhat / (self.buffers["global_variance"] + self.epsilon) ** (1 / 2),
            axis=0,
        )
        grad_x = (
            grad_xhat / (self.buffers["global_variance"] + self.epsilon) ** (1 / 2)
            + 2 / M * grad_var * (x - self.buffers["global_mean"])
            + 1 / M * grad_mean
        )

        grad_gamma = np.sum(output_grad * xhat, axis=0)
        grad_beta = np.sum(output_grad, axis=0)

        grad_dict["gamma"] = grad_gamma
        grad_dict["beta"] = grad_beta
        return grad_x, grad_dict


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
