import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    """
    A SegNet architecture for image segmentation tasks.

    This class implements the SegNet model, which includes an encoder-decoder
    structure with max pooling and unpooling layers. It supports loading pre-trained
    weights from a specified file.
    """

    def __init__(
            self
    ) -> None:
        """
        Initializes the SegNet instance.
        """
        super(SegNet, self).__init__()

        self.enc_conv0 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),

        )
        self.pool0 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2, return_indices=True)  # 256 -> 128

        self.enc_conv1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2, return_indices=True)  # 128 -> 64

        self.enc_conv2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2, return_indices=True)  # 64 -> 32

        self.enc_conv3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2, return_indices=True)  # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, kernel_size=1),

            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=1),
        )

        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(2, stride=2)  # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1),

            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=1),
        )
        self.upsample1 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.upsample2 = nn.MaxUnpool2d(2, stride=2)  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        )
        self.upsample3 = nn.MaxUnpool2d(2, stride=2)  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
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
        mp0_output, mp0_indices = self.pool0(encode_0)

        encode_1 = self.enc_conv1(mp0_output)
        mp1_output, mp1_indices = self.pool1(encode_1)

        encode_2 = self.enc_conv2(mp1_output)
        mp2_output, mp2_indices = self.pool2(encode_2)

        encode_3 = self.enc_conv3(mp2_output)
        mp3_output, mp3_indices = self.pool3(encode_3)

        # bottleneck
        bottleneck = self.bottleneck_conv(mp3_output)

        # decoder
        uncode_0 = self.upsample0(bottleneck, mp3_indices)
        decode_0 = self.dec_conv0(uncode_0)

        uncode_1 = self.upsample1(decode_0, mp2_indices)
        decode_1 = self.dec_conv1(uncode_1)

        uncode_2 = self.upsample2(decode_1, mp1_indices)
        decode_2 = self.dec_conv2(uncode_2)

        uncode_3 = self.upsample3(decode_2, mp0_indices)
        decode_3 = self.dec_conv3(uncode_3)  # no activation

        return decode_3

    def load_model(
            self,
            weights_name: str = "SegNet/SegNet_dice_250e.pt",
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
