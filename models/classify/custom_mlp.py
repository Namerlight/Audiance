import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMLP(torch.nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.Conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.Conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.Conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.Conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)

        self.BatchNorm1 = nn.BatchNorm2d(num_features=8)
        self.BatchNorm2 = nn.BatchNorm2d(num_features=16)
        self.BatchNorm3 = nn.BatchNorm2d(num_features=32)
        self.BatchNorm4 = nn.BatchNorm2d(num_features=64)
        self.BatchNorm5 = nn.BatchNorm2d(num_features=128)

        self.fc = nn.Linear(in_features=51200, out_features=10)

        self.dropout = nn.Dropout(p=0.3, inplace=False)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.Conv2(x)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.Conv3(x)
        x = self.BatchNorm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.Conv4(x)
        x = self.BatchNorm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.Conv5(x)
        x = self.BatchNorm5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x


