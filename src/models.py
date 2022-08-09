import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class NetParent(nn.Module):
    """
    Creates a parent class that has reset_parameters implemented
    so I don't have to re-write it to each child class and can just inherit it
    """

    def __init__(self):
        super(NetParent, self).__init__()

    def forward(self):
        raise NotImplementedError

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)

    def reset_weight(self, layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def reset_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for child in self.children():
            if hasattr(child, 'children'):
                for sublayer in child.children():
                    self.reset_weight(sublayer)
            else:
                self.reset_weight(child)


class ConvBlock(NetParent):
    def __init__(self, n_filters=12, act=nn.Sigmoid()):
        super(ConvBlock, self).__init__()
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=3, padding='same')
        self.conv5 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=5, padding='same')
        self.conv7 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=7, padding='same')
        self.activation = act

    def forward(self, x, ics=None):
        conv3 = torch.max(self.conv3(x), 2)[0]
        conv5 = torch.max(self.conv5(x), 2)[0]
        conv7 = torch.max(self.conv7(x), 2)[0]
        out = torch.cat([conv3, conv5, conv7], 1)
        return out


class LinearBlock(NetParent):
    def __init__(self, n_in, n_hidden, hidden_act=nn.SELU()):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(n_in, n_hidden)
        self.act = hidden_act
        self.out = nn.Linear(n_hidden, 1)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.drop(self.bn1(self.act(self.linear(x))))
        x = F.sigmoid(self.out(x))
        return x


"""
Could introduce a IC weight block here that uses some linear layer
Then in Net could multiply output of Convblock with output of IC_linear block
"""


class Net(NetParent):
    def __init__(self, n_filters, n_hidden, act_cnn=nn.Sigmoid(), act_lin=nn.Sigmoid()):
        super(Net, self).__init__()
        self.conv_block = ConvBlock(n_filters, act=act_cnn)
        self.lin_block = LinearBlock(n_in=3 * n_filters, n_hidden=n_hidden, hidden_act=act_lin)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = self.lin_block(x)

        return x