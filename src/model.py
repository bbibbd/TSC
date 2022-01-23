import torch.nn as nn

class NetworkTSC(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.features=nn.Sequential(
            #in_channels, out_channels, Kernel_size, stride, padding
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace= True),

            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace= True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace= True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*7*7, 1000),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h