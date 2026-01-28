import torch.nn as nn



class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(1, 32),      # input ([batch, 1, 128, 431])
            VGGBlock(32, 64),
            VGGBlock(64, 128)
        )
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 16 * 53, num_classes)  # adjust shape for your input size

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

