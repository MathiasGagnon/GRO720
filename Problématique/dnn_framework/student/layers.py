import numpy as np

from dnn_framework.layer import Layer
epsilon = 1e-5

class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.param = {'w': np.random.randn(output_count, input_count), 'b': np.random.randn(output_count)}

    def get_parameters(self):
        return self.param

    def get_buffers(self):
        return {}

    def forward(self, x):
        return np.matmul(x, self.param['w'].T) + self.param['b'], x

    def backward(self, output_grad, cache):
        return np.dot(output_grad, self.param['w']), {'w':np.dot(output_grad.T, cache), 'b':np.sum(output_grad, axis=0)}


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.input_count = input_count
        self.alpha = alpha
        self.buffer = {'global_mean':np.zeros(input_count), 'global_variance': np.zeros(input_count)}
        self.param = {'gamma': np.zeros(input_count), 'beta': np.zeros(input_count)}

    def get_parameters(self):
        return self.param

    def get_buffers(self):
        return self.buffer

    def forward(self, x):
        if self._is_training:
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)

        if np.all(self.buffer['global_mean'] == 0):
            self.buffer['global_mean'] = batch_mean

        if np.all(self.buffer['global_variance'] == 0):
            self.buffer['global_variance'] = batch_variance

        self.buffer['global_mean'] = (1 - self.alpha) * self.buffer['global_mean'] + self.alpha * batch_mean
        self.buffer['global_variance'] = (1 - self.alpha) * self.buffer['global_variance'] + self.alpha * batch_variance

        return self._forward_evaluation(x)

    def _forward_evaluation(self, x):
        x_hat = ((x - self.buffer['global_mean'])/np.sqrt(self.buffer['global_variance']+ epsilon))
        y = self.param['gamma'] * x_hat + self.param['beta']
        return y, x_hat

    def backward(self, output_grad, cache):
        grad_x_hat = output_grad * self.param['gamma']
        grad_var = np.sum(grad_x_hat * (cache-self.buffer['global_mean']) * -0.5 * (self.buffer['global_variance'] + epsilon)**(-1.5), axis=0)
        grad_mean = -np.sum(grad_x_hat/np.sqrt(self.buffer['global_variance'] + epsilon), axis=0)
        grad_x = (grad_x_hat/np.sqrt(self.buffer['global_variance'])) + (2/self.input_count) * grad_var * (cache-cache-self.buffer['global_mean']) + 1/self.input_count * grad_mean
        return grad_x ,{'gamma': np.sum(output_grad * cache, axis=0), 'beta': np.sum(output_grad, axis=0)}


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        return 1/(1+ np.exp(-x)), x

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
