import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.losses.iloss import ILoss


class FocalLoss(ILoss):
    """
    A class that implements Focal Loss.

    This class inherits from the ILoss interface and provides a method to compute
    the Focal loss for model predictions.

    The Focal loss is defined as:
    """

    def __init__(self) -> None:
        super().__init__()

    def focal_loss(self, y_real, y_pred, eps=1e-8, gamma=2):
        """
        Computes the Focal loss.

        :param y_real: The ground truth labels (real values).
        :type y_real: torch.Tensor
        :param y_pred: The predicted values from the model.
        :type y_pred: torch.Tensor
        :param eps: Small constant to prevent log(0). Default is 1e-8.
        :type eps: float
        :param gamma: Focusing parameter. Default is 2.
        :type gamma: float
        :returns: The computed Focal loss.
        :rtype: torch.Tensor
        """
        bs = y_real.size(0)

        y_pred = y_pred.view(bs, 1, -1)
        y_real = y_real.view(bs, 1, -1)

        y_pred = torch.clamp(torch.sigmoid(y_pred), min=eps, max=1e8)
        loss = - torch.sum(torch.pow(1 - y_pred, gamma) * y_real * torch.log(y_pred) + (1 - y_real) * torch.log(
            torch.clamp(1 - y_pred, min=eps, max=1e8)))
        return loss.mean() / (256 * 256)

    def __call__(self, y_real, y_pred):
        """
        Computes the loss when the instance is called.

        This method acts as a shortcut to compute the Focal loss
        using the provided real and predicted values.

        :param y_real: The ground truth labels (real values).
        :type y_real: torch.Tensor
        :param y_pred: The predicted values from the model.
        :type y_pred: torch.Tensor
        :returns: The computed Focal loss.
        :rtype: torch.Tensor
        """
        return self.focal_loss(y_real, y_pred)