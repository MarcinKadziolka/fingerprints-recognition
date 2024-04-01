import torch
import numpy as np
from torch import nn


class SiameseNetwork(nn.Module):
    def __init__(self, input_dims):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        conv_dims = self.conv_dims(input_dims)
        self.fc = nn.Sequential(
            nn.Linear(conv_dims, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.sigm = nn.Sigmoid()

    def conv_dims(self, input_dims):
        base = torch.zeros(1, *input_dims)
        dims = self.conv(base)
        return int(np.prod(dims.size()))

    def forward(self, input1, input2):
        conv_out1 = self.conv(input1)
        conv_out2 = self.conv(input2)

        if len(conv_out1.size()) == 4:
            flt1 = conv_out1.view(conv_out1.size()[0], -1)
            flt2 = conv_out2.view(conv_out2.size()[0], -1)
        else:
            flt1 = conv_out1.view(-1)
            flt2 = conv_out2.view(-1)
        flt = torch.abs(flt1 - flt2)
        fc_output = self.fc(flt)
        return fc_output.float()

    def save_model(self, path):
        print("... saving model ...")
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        print("... loading model ...")
        self.load_state_dict(torch.load(path))
