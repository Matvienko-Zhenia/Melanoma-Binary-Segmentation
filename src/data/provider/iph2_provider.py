from abc import ABC, abstractmethod
from datasets import Dataset


class IPH2Provider:
    
    @abstractmethod
    def read_data(self):
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> Dataset:
        raise NotImplementedError