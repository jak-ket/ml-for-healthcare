import torch
import torch.nn as nn


class AutoEncoderCnn(torch.nn.Module):
    def __init__(self):
        super(AutoEncoderCnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.fc1 = nn.Linear(5376, 128)
        self.encoder = torch.nn.Sequential(
            self.conv1,
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            self.conv2,
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            self.conv3,
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Flatten(),
            self.fc1,
        )
        self.fc2 = nn.Linear(128, 6144)
        self.deconv1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=0)

        self.decoder = torch.nn.Sequential(
                    self.fc2,
                    torch.nn.ReLU(),
                    torch.nn.Unflatten(1, (256, 24)),
                    self.deconv1,
                    torch.nn.ReLU(),
                    self.deconv2,
                    torch.nn.ReLU(),
                    self.deconv3,
                    
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

