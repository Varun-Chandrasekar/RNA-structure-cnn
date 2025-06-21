# Define CNN model
import torch.nn as nn
import torch.nn.functional as F

class RNACNN(nn.Module):
    def __init__(self):
        super(RNACNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 16, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=4, dilation=4)
        self.fc = nn.Linear(32, 9)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x
