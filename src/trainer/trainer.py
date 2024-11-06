from time import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from torch.optim.lr_scheduler import LRScheduler

from torch.utils.data import DataLoader


class Trainer:

    def train(
            self,
            model: nn.Module,
            opt: Optimizer,
            loss_fn: object,
            epochs: int,
            data_tr: DataLoader,
            path_file: str,
            device: str,
            scheduler: Union[LRScheduler, None] = None,
    ) -> dict:

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

        return model
