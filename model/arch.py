from torchinfo import summary
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_size, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = MNISTClassifier(1, 10)
    summary(model, input_size=(64, 1, 28, 28))