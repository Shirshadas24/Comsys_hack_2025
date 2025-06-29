# model.py
import torch
import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 53 * 53, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.out = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def get_embedding(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        emb1 = self.get_embedding(x1)
        emb2 = self.get_embedding(x2)
        dist = torch.abs(emb1 - emb2)
        out = self.out(dist)
        return out
