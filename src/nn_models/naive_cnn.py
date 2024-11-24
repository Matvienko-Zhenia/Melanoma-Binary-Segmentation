import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.provider.gdrive_weights import WeightsProvider

logger = logging.getLogger('uvicorn.error')
logger.setLevel("INFO")


class NaiveCnn(nn.Module):
    """
    A naive convolutional neural network (CNN) for image segmentation tasks.

    This class implements a simple CNN architecture with multiple convolutional
    layers for feature extraction and a final output layer to generate predictions.
    It also supports loading pre-trained weights from a specified file.

    :param weights_provider: An instance of :py:class:`.WeightsProvider` to manage model weights.
    :type weights_provider: :py:class:`.WeightsProvider`
    """

    def __init__(
            self,
            weights_provider: WeightsProvider,
    ) -> None:
        """
        Initializes the NaiveCnn instance.

        :param weights_provider: An instance of :py:class:`WeightsProvider <src.data.provider.gdrive_weights.WeightsProvider>` to manage model weights.
        :type weights_provider: :py:class:`WeightsProvider <src.data.provider.gdrive_weights.WeightsProvider>`
        """
        super(NaiveCnn).__init__()
        self._weights_provider = weights_provider

        self.enc_conv0 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),

        )

        self.enc_conv1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.enc_conv2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        )

        self.enc_conv3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.final_output = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :type x: torch.Tensor
        :returns: Output tensor after passing through the network.
        :rtype: torch.Tensor
        """
        # encoder
        encode_0 = self.enc_conv0(x)
        encode_1 = self.enc_conv1(encode_0)
        encode_2 = self.enc_conv2(encode_1)
        encode_3 = self.enc_conv3(encode_2)

        final_output = self.final_output(encode_3)

        return final_output

    def load_model(
            self,
            weights_name: str = "NaiveCNN/NaiveCNN_dice_250e.pt",
            device: str = "cpu"
    ) -> None:
        """
        Loads the model weights from the specified file.

        :param weights_name: The file path to the weights file.
        :type weights_name: str
        :param device: The device to load the weights onto ('cpu' or 'cuda').
        :type device: str
        :raises Exception: If there is an error loading the weights.
        """
        try:
            self.load_state_dict(
                torch.load(
                    str(weights_name),
                    map_location=torch.device(device),
                )
            )
        except Exception as e:
            raise Exception

    def load_from_provider(
            self,
            weights_name: str = "NaiveCNN/NaiveCNN_dice_250e.pt",
            device: str = "cpu",
            force_download: bool = False,
    ) -> None:
        """Initialise pytorch trained model

        :param weights_name: string name of model, default value: `NaiveCNN/NaiveCNN_dice_250e.pt`
        :type weights_name: str
        :param device: Device weights and model mapping- `cpu` or `cuda`
        :type device: str
        :param force_download: If True, forces the download even if the file already exists (default is False).
        :type force_download: bool

        :raises Exception: If any

        :returns:
            - None: Inplace method
        """
        try:
            self._weights_provider.download_file(
                file_name=weights_name,
                force=force_download,
            )
            weights_path = self._weights_provider.get_weights_path()

            self.load_state_dict(
                torch.load(
                    str(weights_path / weights_name),
                    map_location=torch.device(device),
                )
            )
        except Exception as e:
            print(e)
            logging.WARNING(f"For {self.__class__.__name__} none weigts found:\n{str(e)}")