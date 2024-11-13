from typing import Any
import torch

from src.metrics.losses.iloss import ILoss


class BCELoss(ILoss):
    """
    A class that implements Binary Cross-Entropy Loss.
    This class inherits from the ILoss interface and provides a method to compute the binary cross-entropy loss for model predictions.
    The binary cross-entropy loss is defined as:
    """

    def __init__(self) -> None:
        super().__init__()

    def bce_loss(self, y_real, y_pred):
        """
        Computes the binary cross-entropy loss.

        :param y_real: The ground truth labels (real values).
        :type y_real: torch.Tensor
        :param y_pred: The predicted values from the model.
        :type y_pred: torch.Tensor
        :returns: The computed binary cross-entropy loss.
        :rtype: torch.Tensor
        """
        return (y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred))).mean()

    def __call__(self, y_real, y_pred):
        """
        Computes the loss when the instance is called.

        This method acts as a shortcut to compute the binary cross-entropy loss
        using the provided real and predicted values.

        :param y_real: The ground truth labels (real values).
        :type y_real: torch.Tensor
        :param y_pred: The predicted values from the model.
        :type y_pred: torch.Tensor
        :returns: The computed binary cross-entropy loss.
        :rtype: torch.Tensor
        """
        return self.bce_loss(y_real, y_pred)