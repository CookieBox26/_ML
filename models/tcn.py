import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm
from collections import OrderedDict


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 以下と同じネットワークを1クラスで実装したもの
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
class TCN(nn.Module):
    def __init__(self,
                 input_size=1,
                 output_size=10,
                 num_channels=[25]*8,
                 kernel_size=7,
                 dropout=0.0):
        super(TCN, self).__init__()
        self.layers = OrderedDict()
        self.num_levels = len(num_channels)

        for i in range(self.num_levels):
            dilation = 2 ** i
            n_in = input_size if (i == 0) else num_channels[i-1]
            n_out = num_channels[i]
            padding = (kernel_size - 1) * dilation
            # ========== TemporalBlock ==========
            self.layers[f'conv1_{i}'] \
                = weight_norm(nn.Conv1d(n_in, n_out, kernel_size,
                                        padding=padding,
                                        dilation=dilation))
            self.layers[f'chomp1_{i}'] = Chomp1d(padding)
            self.layers[f'relu1_{i}'] = nn.ReLU()
            self.layers[f'dropout1_{i}'] = nn.Dropout(dropout)
            self.layers[f'conv2_{i}'] \
                = weight_norm(nn.Conv1d(n_out, n_out, kernel_size,
                                        padding=padding,
                                        dilation=dilation))
            self.layers[f'chomp2_{i}'] = Chomp1d(padding)
            self.layers[f'relu2_{i}'] = nn.ReLU()
            self.layers[f'dropout2_{i}'] = nn.Dropout(dropout)
            self.layers[f'downsample_{i}'] = nn.Conv1d(n_in, n_out, 1) \
                                             if (n_in != n_out) else None
            self.layers[f'relu_{i}'] = nn.ReLU()
            # ===================================
        self.network = nn.Sequential(self.layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_levels):
            self.layers[f'conv1_{i}'].weight.data.normal_(0, 0.01)
            self.layers[f'conv2_{i}'].weight.data.normal_(0, 0.01)
            if self.layers[f'downsample_{i}'] is not None:
                self.layers[f'downsample_{i}'].weight.data.normal_(0, 0.01)
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        for i in range(self.num_levels):
            # Residual Connection
            res = x if (self.layers[f'downsample_{i}'] is None) \
                  else self.layers[f'downsample_{i}'](x)
            out = self.layers[f'conv1_{i}'](x)
            out = self.layers[f'chomp1_{i}'](out)
            out = self.layers[f'relu1_{i}'](out)
            out = self.layers[f'dropout1_{i}'](out)
            out = self.layers[f'conv2_{i}'](out)
            out = self.layers[f'chomp2_{i}'](out)
            out = self.layers[f'relu2_{i}'](out)
            out = self.layers[f'dropout2_{i}'](out)
            x = self.layers[f'relu_{i}'](out + res)
        x = self.linear(x[:, :, -1])
        return F.log_softmax(x, dim=1)
