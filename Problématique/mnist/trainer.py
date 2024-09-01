import os

from tqdm import tqdm

from dnn_framework import Trainer, CrossEntropyLoss, SgdOptimizer, \
    LossMetric, ClassificationAccuracyMetric, LossAccuracyLearningCurves, ClassificationPrecisionMetric
from mnist.dataset import MnistDataset


class MnistTrainer(Trainer):
    def __init__(self, network, learning_rate, epoch_count, batch_size, output_path):
        loss = CrossEntropyLoss()
        optimizer = SgdOptimizer(network.get_parameters(), learning_rate=learning_rate)

        training_dataset = MnistDataset('training')
        validation_dataset = MnistDataset('validation')
        test_dataset = MnistDataset('testing')

        super().__init__(network, training_dataset, validation_dataset, test_dataset,
                         loss, optimizer,
                         epoch_count, batch_size, output_path)

        self._training_loss_metric = LossMetric()
        self._training_accuracy_metric = ClassificationAccuracyMetric()
        self._training_precision_metric = ClassificationPrecisionMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_accuracy_metric = ClassificationAccuracyMetric()
        self._validation_precision_metric = ClassificationPrecisionMetric()
        self._learning_curves = LossAccuracyLearningCurves()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()
        self._training_precision_metric.clear()

    def _measure_training_metrics(self, loss, network_output, target):
        self._training_loss_metric.add(loss.item())
        self._training_accuracy_metric.add(network_output, target)
        self._training_precision_metric.add(network_output, target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()
        self._validation_precision_metric.clear()

    def _measure_validation_metrics(self, loss, network_output, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(network_output, target)
        self._validation_precision_metric.add(network_output, target)

    def _save_figures(self, output_path):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_training_precision_value(self._training_precision_metric.get_precision())
        self._learning_curves.add_validation_precision_value(self._validation_precision_metric.get_precision())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())
        self._learning_curves.save_figure(os.path.join(output_path, 'learning_curves.png'))

    def _print_metrics(self):
        print('\nTraining : Loss={}, Accuracy={} Precision={}'.format(self._training_loss_metric.get_loss(),
                                                         self._training_accuracy_metric.get_accuracy(), 
                                                         self._training_precision_metric.get_precision()))
        print('Validation : Loss={}, Accuracy={} Precision={}\n'.format(self._validation_loss_metric.get_loss(),
                                                           self._validation_accuracy_metric.get_accuracy(), 
                                                           self._validation_precision_metric.get_precision()))

    def _test(self, network, test_dataset_loader):
        test_accuracy_metric = ClassificationAccuracyMetric()
        test_precision_metric = ClassificationPrecisionMetric()

        for x, target in tqdm(test_dataset_loader):
            y = network.forward(x)
            test_accuracy_metric.add(y, target)
            test_precision_metric.add(y, target)

        print('Accuracy={} Precision={}'.format(test_accuracy_metric.get_accuracy(), test_precision_metric.get_precision()))
