import torch
import torch.nn as nn


class AB(nn.Module):

    def __init__(self, input_size=16, hidden_size=16,  num_classes=10) -> None:
        super(AB, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return 1