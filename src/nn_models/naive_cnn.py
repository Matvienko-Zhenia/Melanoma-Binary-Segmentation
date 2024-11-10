import torch
import torch.nn as nn


class NaiveCnn(nn.Module):
    """
    A naive convolutional neural network (CNN) for image segmentation tasks.

    This class implements a simple CNN architecture with multiple convolutional
    layers for feature extraction and a final output layer to generate predictions.
    It also supports loading pre-trained weights from a specified file.
    """

    def __init__(
            self
    ) -> None:
        """
        Initializes the NaiveCnn instance.
        """
        super(NaiveCnn).__init__()

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