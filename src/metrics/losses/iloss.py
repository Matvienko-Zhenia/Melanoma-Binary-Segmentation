from abc import ABC, abstractmethod
from typing import Any


class ILoss(ABC):

    @abstractmethod
    def __call__(self, y_real, y_pred):
        raise NotImplemented