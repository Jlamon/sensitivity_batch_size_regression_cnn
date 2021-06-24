import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # x = self.conv2_drop(x)

        # Layer 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Flatten
        x = x.view(-1, 320)

        # Dense
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
