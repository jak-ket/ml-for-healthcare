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
        self.deconv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=4, padding=0)
        self.deconv2 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=4, padding=0)
        self.deconv3 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=4, padding=1) 
        self.decoder = nn.Sequential(
            nn.ReLU(),
            torch.nn.Unflatten(1, (1, 128)),    
            self.deconv1,
            nn.ReLU(),
            nn.Upsample(scale_factor=1.24),  
            self.deconv2,
            nn.ReLU(),
            nn.Upsample(scale_factor=1.24),
            self.deconv3
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

