from abc import ABC
from collections import OrderedDict
from typing import Union
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class NetParent(nn.Module):
    """
    Mostly a QOL superclass
    Creates a parent class that has reset_parameters implemented and .device
    so I don't have to re-write it to each child class and can just inherit it
    """

    def __init__(self):
        super(NetParent, self).__init__()
        # device is cpu by default
        self.device = 'cpu'

    def forward(self):
        raise NotImplementedError

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)

    @staticmethod
    def reset_weight(layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def reset_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for child in self.children():
            if hasattr(child, 'children'):
                for sublayer in child.children():
                    self.reset_weight(sublayer)
            if hasattr(child, 'reset_parameters'):
                self.reset_weight(child)

    def to(self, device):
        # Work around, so we can get model.device for all NetParent
        #
        super(NetParent, self).to(device)
        self.device = device


class Standardizer(nn.Module):
    def __init__(self):
        super(Standardizer, self).__init__()
        self.mu = 0
        self.sigma = 1
        self.fitted = False

    def fit(self, x_train):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        self.mu = x_train.mean(axis=0)
        self.sigma = x_train.std(axis=0)
        # Fix issues with sigma=0 that would cause a division by 0 and return NaNs
        self.sigma[torch.where(self.sigma==0)] = 1e-12
        self.fitted = True

    def forward(self, x):
        assert self.fitted, 'Standardizer has not been fitted. Please fit to x_train'
        return (x - self.mu) / self.sigma

    def reset_parameters(self, **kwargs):
        self.mu = 0
        self.sigma = 0
        self.fitted = False


class ConvBlock(NetParent):
    def __init__(self, input_length=12, n_filters=10,  # maxpool_ks = 4, n_embedded = 30,
                 act=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=3, padding='same')
        self.conv5 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=5, padding='same')
        self.conv7 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=7, padding='same')
        # self.maxpool = nn.MaxPool1d(kernel_size=maxpool_ks)
        self.act = act
        # 3* n_filters because 3 conv layers
        # FC_out here may be unecessary when using torch.max instead of maxpool with kernel size of 4
        # self.fc_out = nn.Linear(3 * n_filters, n_embedded)

    def reshape_input(self, x):
        in_channels = self.conv3.in_channels
        assert (len(x.shape) == 3 and x.shape[
            -1] == in_channels), f'Provided input of shape {x.shape} has the wrong dimensions.\n' \
                                 f'It should have 3 dimensions and have in_channels={in_channels} for the last dimension.'
        if type(x) == np.ndarray:
            return torch.from_numpy(np.transpose(x, [0, 2, 1])).to(self.conv3.weight.device)
        elif type(torch.Tensor):
            return torch.permute(x, [0, 2, 1])

    def forward(self, x):
        x = self.reshape_input(x)
        conv3 = torch.max(self.act(self.conv3(x)), 2)[0]
        conv5 = torch.max(self.act(self.conv5(x)), 2)[0]
        conv7 = torch.max(self.act(self.conv7(x)), 2)[0]
        out = torch.cat([conv3, conv5, conv7], 1)  # .flatten(start_dim=1, end_dim=2)
        # out = self.fc_out(out)
        return out


class DICNN(NetParent):

    def __init__(self, input_length=12, n_filters=10,  # maxpool_ks=4, n_embedded = 30,
                 n_hidden=32, n_props=10, act=nn.ReLU()):
        super(DICNN, self).__init__()
        self.conv_block = ConvBlock(n_filters)
        self.act = act
        self.dropout = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(n_hidden)
        self.fc_in = nn.Linear(2 * 3 * n_filters + n_props, n_hidden)
        self.fc_out = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.input_length = input_length

    def forward(self, x_mut, x_wt, x_props):
        conv_mut = self.conv_block(x_mut)
        conv_wt = self.conv_block(x_wt)
        # Concat convolution outputs and properties and feed into FC layer
        x = torch.cat([conv_mut, conv_wt, x_props], dim=1)
        x = self.dropout(self.batchnorm(self.act(self.fc_in(x))))
        out = self.sigmoid(self.fc_out(x))
        return out


class FFN(NetParent):
    def __init__(self, n_in=21, n_hidden=32, n_layers=1, act=nn.ReLU(), dropout=0.0):
        super(FFN, self).__init__()
        self.in_layer = nn.Linear(n_in, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = act
        hidden_layers = [nn.Linear(n_hidden, n_hidden), self.dropout, self.activation] * n_layers
        self.hidden = nn.Sequential(*hidden_layers)
        # Either use Softmax with 2D output or Sigmoid with 1D output
        self.out_layer = nn.Linear(n_hidden, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.flatten(start_dim=1, end_dim=2)
        x = self.activation(self.in_layer(x))
        x = self.hidden(x)
        out = F.sigmoid(self.out_layer(x))
        return out


class FFNetPipeline(NetParent):
    def __init__(self, n_in=21, n_hidden=32, n_layers=1, act=nn.ReLU(), dropout=0.3):
        super(FFNetPipeline, self).__init__()
        self.standardizer = Standardizer()
        self.input_length = n_in
        self.ffn = FFN(n_in, n_hidden, n_layers, act, dropout)

    def forward(self, x):
        # Need to do self.standardizer.fit() somewhere in the nested_kcv function
        x = self.standardizer(x)
        x = self.ffn(x)
        return x

    def fit_standardizer(self, x):
        assert self.training, 'Must be in training mode to fit!'
        self.standardizer.fit(x)

    def reset_parameters(self, **kwargs):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                try:
                    child.reset_parameters(**kwargs)
                except:
                    print('here xd', child)


class CNNetPipeline(NetParent):
    def __init__(self, input_length=12, n_filters=10, n_hidden=32, n_props=14, act=nn.ReLU()):
        super(CNNetPipeline, self).__init__()
        self.standardizer = Standardizer()
        self.input_length = input_length
        self.n_props = n_props
        self.dicnn = DICNN(input_length, n_filters, n_hidden, n_props, act)

    def _extract_reshape_input(self, x):
        x_mut = x[:, :self.input_length * 20].view(-1, self.input_length, 20).view(-1, self.input_length, 20)
        x_wt = x[:, self.input_length * 20: 2 * self.input_length * 20].view(-1, self.input_length, 20)
        x_props = x[:, -self.n_props:]
        return x_mut, x_wt, x_props

    def forward(self, x):
        # Takes concatenated X as input to make it easier in the
        # Feature processing functions
        x_mut, x_wt, x_props = self._extract_reshape_input(x)
        # Need to do self.standardizer.fit() somewhere in the nested_kcv function
        # Somewhere before calling train_loop()
        x_props = self.standardizer(x_props)
        output = self.dicnn(x_mut, x_wt, x_props)
        return output

    def reset_parameters(self, **kwargs):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                try:
                    child.reset_parameters(**kwargs)
                except:
                    print('here xd', child)


"""
Old CNN/FFN stuff for NN_freq training
"""


class ConvBlockOld(NetParent):
    def __init__(self, n_filters=12, act=nn.Sigmoid()):
        super(ConvBlockOld, self).__init__()
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=3, padding='same')
        self.conv5 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=5, padding='same')
        self.conv7 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=7, padding='same')
        self.activation = act

    def reshape_input(self, x):
        in_channels = self.conv3.in_channels
        assert (len(x.shape) == 3 and x.shape[
            -1] == in_channels), f'Provided input of shape {x.shape} has the wrong dimensions.\'' \
                                 f'It should have 3 dimensions and have in_channels={in_channels} for the last dimension.'
        if type(x) == np.ndarray:
            return torch.from_numpy(np.transpose(x, [0, 2, 1])).to(self.conv3.weight.device)
        elif type(torch.Tensor):
            return torch.permute(x, [0, 2, 1])

    def forward(self, x, ics=None):
        x = self.reshape_input(x)
        conv3 = torch.max(self.conv3(x), 2)[0]
        conv5 = torch.max(self.conv5(x), 2)[0]
        conv7 = torch.max(self.conv7(x), 2)[0]
        out = torch.cat([conv3, conv5, conv7], 1)
        return out

    def load_convblock(self, path):
        """
        Reloads a convblock weights and sets to eval (to be used for mixed models)
        Args:
            path:

        Returns:

        """
        state_dict = torch.load(path)
        conv_keys = [x for x in state_dict.keys() if 'conv_block' in x or 'conv' in x]
        if conv_keys[0].startswith('conv_block.'):
            conv_keys_stripped = [x.lstrip('conv_block').lstrip('.') for x in conv_keys]
        conv_states = OrderedDict(
            (key_strip, state_dict[key]) for (key_strip, key) in zip(conv_keys_stripped, conv_keys))
        self.load_state_dict(conv_states)
        self.eval()


class LinearBlock(NetParent):
    def __init__(self, n_in, n_hidden, hidden_act=nn.SELU()):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(n_in, n_hidden)
        self.act = hidden_act
        self.out = nn.Linear(n_hidden, 1)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        if len(x.shape == 3):
            x = x.flatten(start_dim=1, end_dim=2)
        x = self.drop(self.bn1(self.act(self.linear(x))))
        x = F.sigmoid(self.out(x))
        return x


"""
Could introduce a IC weight block here that uses some linear layer
Then in Net could multiply output of Convblock with output of IC_linear block
"""


class Net(NetParent):
    def __init__(self, n_filters, n_hidden, add_rank=False, act_cnn=nn.Sigmoid(), act_lin=nn.Sigmoid()):
        super(Net, self).__init__()
        # IGNORE THIS FOR NOW
        # TODO: ADD IMPLEMENTATION TO ADD RANK AS INPUT TO NN
        # self.rank = add_rank
        # if add_rank:
        #     n_hidden = n_hidden + 1 # Add one extra node for rank
        self.conv_block = ConvBlock(n_filters, act=act_cnn)
        self.lin_block = LinearBlock(n_in=3 * n_filters, n_hidden=n_hidden, hidden_act=act_lin)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = self.lin_block(x)

        return x


"""
    Here put in the mixed models ?
"""


class MixedTreesNet(RandomForestClassifier):
    def __init__(self, n_filters, activation, conv_weights_path,
                 TreeBasedModel: Union[RandomForestClassifier, XGBClassifier]):
        super(MixedTreesNet, self).__init__()
        # Here load weights in convblock
        self.conv_block = ConvBlock(n_filters, activation)
        self.conv_block.load_convblock(conv_weights_path)
        self.conv_block.eval()
        # Make a clone of the tree based model
        self.trees = sklearn.base.clone(TreeBasedModel)

    def convo(self, x):
        with torch.no_grad():
            x = self.conv_block(x)
            x = x.detach().cpu().numpy()
        return x

    def fit(self, x, y, sample_weight=None):
        # Here self.conv_block should reshape the input automatically
        # Then run the convblock and convert/detach send to cpu
        x_feat = self.convo(x)
        self.trees.fit(x_feat, y, sample_weight)

    def predict(self, x):
        x_feat = self.convo(x)
        y_pred = self.trees.predict(x_feat)
        return y_pred

    def predict_proba(self, x):
        x_feat = self.convo(x)
        y_proba = self.trees.predict_proba(x_feat)
        return y_proba

    def predict_log_proba(self, x):
        if hasattr(self.trees, 'predict_log_proba'):
            x_feat = self.convo(x)
            y_log_proba = self.trees.predict_log_proba(x_feat)
            return y_log_proba
        else:
            raise NotImplementedError

    def get_params(self, deep=True):
        return self.trees.get_params()
