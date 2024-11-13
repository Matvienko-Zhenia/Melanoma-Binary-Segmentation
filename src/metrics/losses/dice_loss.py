import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss:
    """
    A class that implements Dice Loss.

    This class provides a method to compute
    the Dice loss for model predictions.

    The Dice loss is defined as:
    """

    def __init__(self) -> None:
        pass

    def dice_loss(self, y_real, y_pred):
        """
        Computes the Dice loss.

        :param y_real: The ground truth labels (real values).
        :type y_real: torch.Tensor
        :param y_pred: The predicted values from the model.
        :type y_pred: torch.Tensor
        :returns: The computed Dice loss.
        :rtype: torch.Tensor
        """
        bs = y_real.size(0)
        y_pred = y_pred.view(bs, 1, -1)
        y_real = y_real.view(bs, 1, -1)

        y_pred = F.logsigmoid(y_pred).exp()

        num = 2 * torch.sum(y_real * y_pred)
        den = torch.sum(y_real + y_pred) + 1e-9
        res = 1 - num / den  # / (256 * 256)
        return res.mean()

    def __call__(self, y_real, y_pred):
        """
        Computes the loss when the instance is called.

        This method acts as a shortcut to compute the Dice loss
        using the provided real and predicted values.

        :param y_real: The ground truth labels (real values).
        :type y_real: torch.Tensor
        :param y_pred: The predicted values from the model.
        :type y_pred: torch.Tensor
        :returns: The computed Dice loss.
        :rtype: torch.Tensor
        """
        return self.dice_loss(y_real, y_pred)