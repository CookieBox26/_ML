import torch
import torch.nn.functional as F
import torch.nn as nn


# ラップしたGRUクラス
class GRU(nn.Module):

    def __init__(self,
                 input_size=1,    # Sequential MNIST タスクなら入力は1次元
                 output_size=10,  # Sequential MNIST タスクなら出力は10次元
                 num_layers=1,    # GRU ブロックの積み重ね数
                 d_hidden=128,    # GRU ブロックの出力次元数（隠れ状態の次元数）
                 initial_update_gate_bias=0.5,  # 更新ゲートのバイアスの初期値
                 dropout=0.0):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.d_hidden = d_hidden
        self.gru = nn.GRU(input_size=input_size, hidden_size=d_hidden,
                          num_layers=num_layers, dropout=dropout,
                          batch_first=True)
        self.linear = nn.Linear(d_hidden, output_size)
        self.init_weights(initial_update_gate_bias)

    # 更新ゲートのバイアスの初期値をセット
    def init_weights(self, initial_update_gate_bias):    
        for i_layer in range(self.num_layers):
            bias = getattr(self.gru, f'bias_ih_l{i_layer}')
            bias.data[self.d_hidden:(2*self.d_hidden)] = initial_update_gate_bias
            bias = getattr(self.gru, f'bias_hh_l{i_layer}')
            bias.data[self.d_hidden:(2*self.d_hidden)] = initial_update_gate_bias
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        x = self.linear(hidden[-1, :, :])
        return F.log_softmax(x, dim=1), hidden

    # バッチサイズを渡すと出力特徴の初期テンソルをつくってくれる
    def generate_initial_hidden(self, batch_size):
        return torch.zeros([self.num_layers, batch_size, self.d_hidden])
