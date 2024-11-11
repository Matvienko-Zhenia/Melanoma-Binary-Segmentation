from time import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torch.optim.lr_scheduler import LRScheduler

from torch.utils.data import DataLoader


class Trainer:
    """
    A class for training, validating, and evaluating machine learning models.

    This class encapsulates the functionality needed to train a neural network
    model, validate its performance on a separate dataset, and evaluate it using
    custom metrics. It also includes methods for making predictions with the model.

    """

    def train(
            self,
            model: nn.Module,
            opt: Optimizer,
            loss_fn: object,
            epochs: int,
            data_tr: DataLoader,
            data_val: DataLoader,
            path_file: str,
            device: str,
            scheduler: Union[LRScheduler, None] = None,
    ) -> dict:
        """
        Trains the model for a specified number of epochs.

        :param model: The model to be trained.
        :type model: nn.Module
        :param opt: The optimizer used for training the model.
        :type opt: Optimizer
        :param loss_fn: The loss function to compute the training loss.
        :type loss_fn: object
        :param epochs: The number of epochs to train the model.
        :type epochs: int
        :param data_tr: DataLoader for the training dataset.
        :type data_tr: DataLoader
        :param data_val: DataLoader for the validation dataset.
        :type data_val: DataLoader
        :param path_file: The path to save the best model weights.
        :type path_file: str
        :param device: The device to perform computations on (e.g., "cpu" or "cuda").
        :type device: str
        :param scheduler: Learning rate scheduler (optional).
        :type scheduler: Union[LRScheduler, None]
        :returns: A dictionary containing training and validation losses and the model.
        :rtype: dict
        """
        epochs_list = []
        loss_train = []
        loss_valid = []

        global_loss = float("inf")

        for epoch in range(epochs):
            tic = time()
            print('* Epoch %d/%d' % (epoch + 1, epochs))

            avg_loss = 0
            model.train()  # train mode
            for X_batch, Y_batch in data_tr:
                # data to device
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                # set parameter gradients to zero
                opt.zero_grad()
                # forward
                Y_pred = model(X_batch)
                loss = loss_fn(Y_batch, Y_pred)  # forward-pass
                loss.backward()  # backward-pass
                opt.step()  # update weights

                # calculate loss to show the user
                avg_loss += loss / len(data_tr)

            if scheduler:
                scheduler.step()
            toc = time()
            print('loss: %f' % avg_loss)

            current_loss_train = avg_loss.detach().cpu().numpy().tolist()
            current_loss_valid = self.validation(
                model=model,
                data_val=data_val,
                loss_fn=loss_fn,
                device=device,
            )

            loss_train += [current_loss_train]
            loss_valid += [current_loss_valid]
            epochs_list += [epoch]

            # show intermediate results
            model.eval()  # testing mode

            if current_loss_valid <= global_loss:
                global_loss = current_loss_valid
                torch.save(model.state_dict(), path_file)

        return {
            "loss_train": loss_train,
            "loss_valid": loss_valid,
            "epochs_list": epochs_list,
            "model": model,
        }

    def validation(
            self,
            model: nn.Module,
            data_val: DataLoader,
            loss_fn: object,
            device: str,
    ) -> np.array:
        """
        Validates the model on the validation dataset.

        :param model: The model to be validated.
        :type model: nn.Module
        :param data_val: DataLoader for the validation dataset.
        :type data_val: DataLoader
        :param loss_fn: The loss function to compute validation loss.
        :type loss_fn: object
        :param device: The device to perform computations on.
        :type device: str
        :returns: The average validation loss.
        :rtype: np.array
        """
        loss = []

        model.eval()
        with torch.no_grad():
            for X_batch, Y_batch in data_val:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                Y_pred = model(X_batch)
                loss_value = loss_fn(Y_batch, Y_pred)

                loss += [loss_value.detach().cpu().numpy().tolist()]

        loss = np.array(loss).mean()
        return loss

    def predict(self, model, data):
        """
        Makes predictions using the trained model.

        :param model: The trained model to use for predictions.
        :type model: nn.Module
        :param data: DataLoader containing the input data for predictions.
        :type data: DataLoader
        :returns: An array of predictions.
        :rtype: np.array
        """
        model.eval()
        Y_pred = [X_batch for X_batch, _ in data]
        return np.array(Y_pred)

    def score_model(
            self,
            model: nn.Module,
            metric: object,
            data: DataLoader,
            device: str,
    ):
        """
        Evaluates the model using a specified metric.

        :param model: The model to be evaluated.
        :type model: nn.Module
        :param metric: The metric function to use for evaluation.
        :type metric: object
        :param data: DataLoader for the dataset to evaluate on.
        :type data: DataLoader
        :param device: The device to perform computations on.
        :type device: str
        :returns: The average score computed using the metric.
        :rtype: float
        """
        model.eval()
        scores = 0
        for X_batch, Y_label in data:
            Y_pred = (torch.sigmoid(model(X_batch.to(device))) > 0.5).to(torch.float32)
            scores += metric(Y_pred, Y_label.to(device)).mean().item()

        return scores / len(data)