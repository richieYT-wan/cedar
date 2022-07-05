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

    def reset_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for layer in self.children():
            layer.zero_grad()
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.reset_parameters()
                continue
            if hasattr(layer, 'weight'):
                # re-init
                torch.nn.init.xavier_uniform(layer.weight)
            if hasattr(layer, 'bias'):
                torch.nn.init.zeros_(layer.bias)


################ ARCHITECTURES #####################
class CNN_1(NetParent):
    def __init__(self, input_length, n_filters, n_hidden, k, act=nn.ReLU(), p_drop=0.33, drop_bn=False):
        super(CNN_1, self).__init__()
        if input_length + 1 - k <= 0:
            raise ValueError(f"The kernel size {k} provided won't work!\n"
                             f"input_length+1-k = {input_length - k + 1}. "
                             "Please ensure that input_length+1-k > 0.")
        self.conv_1 = nn.Conv1d(in_channels=21,
                                out_channels=n_filters,
                                kernel_size=k, stride=1, padding=0)
        self.fc1 = nn.Linear(n_filters, n_hidden)
        self.maxpool = nn.MaxPool1d(kernel_size=input_length - k + 1,
                                    stride=None)
        self.fc_out = nn.Linear(n_hidden, 1)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.drop = nn.Dropout(p=p_drop)
        self.drop_bn = drop_bn
        self.act = act
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.act(self.conv_1(x))
        x, _ = torch.max(x, axis=2)
        x = torch.squeeze(x)
        # FC(x) -> activation -> batchnorm -> dropout
        if self.drop_bn == True:
            x = self.drop(self.bn(self.act(self.fc1(x))))
        else:
            x = self.act(self.fc1(x))
        out = self.act_out(self.fc_out(x))

        return out


class FFN_1(NetParent):
    def __init__(self, nh_1, n_layers, act=nn.ReLU(), p_drop=0.33):
        super(FFN_1, self).__init__()

        self.drop = nn.Dropout(p_drop)
        self.fc_in = nn.Linear(9 * 21, nh_1)
        self.bn_in = nn.BatchNorm1d(nh_1)
        layers = []
        nh = nh_1
        self.act = act
        for n in range(n_layers):
            layers.extend([nn.Linear(nh, nh // 2), self.act, nn.BatchNorm1d(nh // 2), self.drop])
            nh = nh // 2
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(nh, 1)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        # Reshape from [N, 21, 9] to [N, 189] for input layer
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.drop(self.bn_in(self.act(self.fc_in(x))))
        # nn sequential layers
        x = self.layers(x)
        # output
        x = self.act_out(self.fc_out(x))
        return x
