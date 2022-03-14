import torch.nn as nn

class ResBlockConv(nn.Module):

    def __init__(self, in_out_channel, hidden_channel,
                 kernel_size=3, stride=1, padding=1):
        super(ResBlockConv, self).__init__()
        self.conv1 = nn.Conv2d(in_out_channel, hidden_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = nn.Conv2d(hidden_channel, in_out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.bn2 = nn.BatchNorm2d(in_out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(x + out)
        return out

class ResBlockConvTranspose(nn.Module):

    def __init__(self, in_out_channel, hidden_channel,
                 kernel_size=3, stride=1, padding=1):
        super(ResBlockConvTranspose, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_out_channel, hidden_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = nn.ConvTranspose2d(hidden_channel, in_out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.bn2 = nn.BatchNorm2d(in_out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(x + out)
        return out