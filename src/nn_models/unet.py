import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.provider.gdrive_weights import WeightsProvider


class UNet(nn.Module):
    """  
    A U-Net architecture for image segmentation tasks.

    This class implements the U-Net model, which consists of a contracting (encoder)
    path and an expansive (decoder) path. The architecture allows for high-resolution 
    feature extraction and reconstruction, making it suitable for tasks like biomedical 
    image segmentation.

    :param weights_provider: An instance of :py:class:`.src.data.provider.gdrive_weights` to manage model weights.
    :type weights_provider: :py:class:`.src.data.provider.gdrive_weights`  
    """  
    def __init__(
            self,
            weights_provider: WeightsProvider,
        ) -> None:
        """  
        Initializes the UNet instance.

        :param weights_provider: An instance of :py:class:`.src.data.provider.gdrive_weights` to manage model weights.
        :type weights_provider: :py:class:`.src.data.provider.gdrive_weights`  
        """  
        super(UNet, self).__init__()
        self._weights_provider = weights_provider

        self.enc_conv0 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),

        )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 256 -> 128

        self.enc_conv1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3),

        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 128 -> 64

        self.enc_conv2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3),

        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 64 -> 32

        self.enc_conv3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=3),

        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 32 -> 16

        self.enc_conv4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3),

        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 16 -> 8

        self.enc_conv5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=3),

        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 8 -> 4

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 4 -> 8
        self.dec_conv0 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

        )

        # decoder (upsampling)
        self.upsample1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 8 -> 16
        self.dec_conv1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),

        )

        # decoder (upsampling)
        self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # 16 -> 32
        self.dec_conv2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),

        )

        self.upsample3 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)  # 32 -> 64
        self.dec_conv3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),

        )

        self.upsample4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # 64 -> 128
        self.dec_conv4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),

        )

        self.upsample5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 128 -> 256
        self.dec_conv5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=3),

        )

    def forward(self, x):
        """  
        Defines the forward pass of the model.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :type x: torch.Tensor  
        :returns: Output tensor after passing through the network.
        :rtype: torch.Tensor  
        """  
        x = F.pad(x, (4, 4, 4, 4), mode='reflect')

        # encoder
        e0 = self.enc_conv0(x)
        d0, _ = self.pool0(e0)

        e1 = self.enc_conv1(d0)
        d1, _ = self.pool1(e1)

        e2 = self.enc_conv2(d1)
        d2, _ = self.pool2(e2)

        e3 = self.enc_conv3(d2)
        d3, _ = self.pool3(e3)

        e4 = self.enc_conv4(d3)
        d4, _ = self.pool4(e4)

        e5 = self.enc_conv5(d4)
        d5, _ = self.pool5(e5)

        # bottleneck
        b = self.bottleneck_conv(d5)

        # decoder
        u0 = self.upsample0(b)
        u0 = torch.cat((u0, F.pad(F.pad(e5, (1, 2, 2, 1)), (-8,-10,-10, -8))), axis=1)
        d0 = self.dec_conv0(u0)

        u1 = self.upsample1(d0)
        u1 = torch.cat((u1, F.pad(F.pad(e4, (1, 2, 2, 1)), (-8, -10, -10, -8))), axis=1)
        d1 = self.dec_conv1(u1)

        u2 = self.upsample2(d1)
        u2 = torch.cat((u2, F.pad(F.pad(e3, (2, 1, 1, 2)),(-10, -8, -8, -10))), axis=1)
        d2 = self.dec_conv2(u2)

        u3 = self.upsample3(d2)
        u3 = torch.cat((u3, F.pad(F.pad(e2, (1, 1, 1, 1)), (-8, -8, -8, -8))), axis=1)
        d3 = self.dec_conv3(u3)

        u4 = self.upsample4(d3)
        u4 = torch.cat((u4, F.pad(F.pad(e1, (1, 2, 2, 1)), (-8, -8, -8, -8))), axis=1)
        d4 = self.dec_conv4(u4)

        u5 = self.upsample5(d4)
        u5 = torch.cat((u5, F.pad(e0, (-5, -5, -5, -5))), axis=1)
        d5 = self.dec_conv5(u5)

        return d5

    def load_model(
            self,
            weights_name: str = "UNet/UNet_dice_250e.pt",
            device: str = "cpu"
        ) -> None:

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
            weights_name: str = "UNet/UNet_dice_250e.pt",
            device: str = "cpu",
            force_download: bool = False,  
        ) -> None:
        """Initialise pytorch trained model

        :param weights_name: string name of model, default value: `UNet/UNet_dice_250e.pt`
        :type weights_name: str
        :param device: Device weights and model mapping- `cpu` or `cuda`
        :type device: str
        :param force_download: Operation to run against a and b
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