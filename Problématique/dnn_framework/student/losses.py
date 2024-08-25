import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        x_softmax = softmax(x)

        n_samples = x.shape[0]
        n_classes = x.shape[1]

        target_samples = np.zeros(x.shape)

        for sample_index in range(n_samples):
            for class_index in range(n_classes):
                if target[sample_index] == class_index:
                    target_samples[sample_index, class_index] = 1
                else:
                    target_samples[sample_index, class_index] = 0
        
        losses = []
        for sample_index in range(n_samples):
            losses.append(-1*np.sum(target_samples[sample_index] * np.log(x_softmax[sample_index])))

        loss = np.mean(losses)

        input_grad = (x_softmax - target_samples) / n_samples

        return loss, input_grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    softmax_values = []
    for sample_index in range(x.shape[0]):
        x_i = x[sample_index]
        softmax_values.append(np.exp(x_i)/np.sum(np.exp(x_i)))

    softmax_values = np.array(softmax_values)
    return softmax_values


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        n = 1
        for dimension_size in list(x.shape):
            n = n * dimension_size

        loss = (np.sum(((x-target)**2)))/n

        input_grad = (2/n)*(x-target)

        return loss, input_grad
