from pathlib import Path
from datasets import Dataset
import os

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from torch.utils.data import DataLoader

from src.data.provider.iph2_provider import IPH2Provider


class PH2Provider(IPH2Provider):
    """  
    A provider class for handling the PH2 dataset.  
   
    :ivar dataset_path: The path to the dataset directory.  
    :vartype dataset_path: str  
    :ivar size: The desired size of the images.  
    :vartype size: tuple  
    """  
    def __init__(
            self,
            dataset_path: str,
            size: tuple = (256, 256),
        ):
        """  
        Constructs all the necessary attributes for the PH2Provider object.  
  
        :param dataset_path: The path to the dataset directory.  
        :type dataset_path: str  
        :param size: The desired size of the images for train and inference.  
        :type size: tuple  
        """  
        super().__init__()
        self._dataset_path = dataset_path
        self._size = size

    def read_data(self):
        """  
        Reads images and lesions from the dataset path.  
  
        :returns: Tuple containing images and lesions.  
        :rtype: tuple  
        """  
        images = []
        lesions = []
        for root, dirs, files in os.walk(self._dataset_path):
            if root.endswith('_Dermoscopic_Image'):
                images.append(imread(os.path.join(root, files[0])))
            if root.endswith('_lesion'):
                lesions.append(imread(os.path.join(root, files[0])))

        X = [resize(x, self._size, mode='constant', anti_aliasing=True,) for x in images]
        Y = [resize(y, self._size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]

        X = np.array(X, np.float32)
        Y = np.array(Y, np.float32)

        return X, Y

    def train_test_split(self, X, Y) -> tuple:
        """  
        Splits the dataset into training, validation, and test sets.  
  
        :param X: Input data.  
        :type X: np.ndarray  
        :param Y: Target data.  
        :type Y: np.ndarray  
        :returns: Tuple of train, validation, and test indices.  
        :rtype: tuple  
        """ 
        ix = np.random.choice(len(X), len(X), False)
        train, validation, test = np.split(ix, [100, 150])

        return train, validation, test
    
    def get_data(self, batch_size: int = 25) -> list[Dataset]:
        """  
        Prepares the data loaders for training, validation, and testing.  
  
        :param batch_size: The number of samples per batch.  
        :type batch_size: int  
        :returns: Tuple containing data loaders for train, validation, and test sets.  
        :rtype: tuple  
        """      
        X, Y = self.read_data()
        train, validation, test = self.train_test_split(X, Y)

        data_train = DataLoader(list(zip(np.rollaxis(X[train], 3, 1), Y[train, np.newaxis])), 
                     batch_size=batch_size, shuffle=True)
        data_validation = DataLoader(list(zip(np.rollaxis(X[validation], 3, 1), Y[validation, np.newaxis])),
                            batch_size=batch_size, shuffle=True)
        data_test = DataLoader(list(zip(np.rollaxis(X[test], 3, 1), Y[test, np.newaxis])),
                            batch_size=batch_size, shuffle=True)
        
        return (data_train, data_validation, data_test)