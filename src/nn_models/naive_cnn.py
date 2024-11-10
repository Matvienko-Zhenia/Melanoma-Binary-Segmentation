import torch
import torch.nn as nn




class NaiveCnn(nn.Module):
    def __init__(
            self,
    ) -> None:
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
        try:
            self.load_state_dict(
                torch.load(
                    str(weights_name),
                    map_location=torch.device(device),
                )
            )
        except Exception as e:
            raise Exception