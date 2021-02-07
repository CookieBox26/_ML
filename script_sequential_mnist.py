from utils.data_loader import MNIST
from models.gru import GRU
from models.tcn import TCN
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


# 1エポック学習します
def train(model, optimizer, train_loader, log_interval=10):
    model.train()
    loss_in_log_interval = 0
    n_samples_processed = 0
    for i_batch, (x, y) in enumerate(train_loader):
        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()
        if type(model) is TCN:
            y_hat = model(x)
        else:
            hidden = model.generate_initial_hidden(x.size()[0])
            y_hat, hidden = model(x, hidden)
        loss = F.nll_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        loss_in_log_interval += loss.item()
        n_samples_processed += x.size()[0]
        if (i_batch + 1) % log_interval == 0:
            print('{}/{} バッチ ({}/{} サンプル) 流れました  最近 {} バッチの平均損失 {}'.format(
                i_batch + 1, len(train_loader),
                n_samples_processed, len(train_loader.dataset),
                log_interval, loss_in_log_interval / log_interval
            ))
            loss_in_log_interval = 0


# テストデータに対してテストします
def test(model, test_loader):
    model.eval()
    n_total = len(test_loader.dataset)
    test_loss = 0
    n_correct = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            if type(model) is TCN:
                y_hat = model(x)
            else:
                hidden = model.generate_initial_hidden(x.size()[0])
                y_hat, hidden = model(x, hidden)
            test_loss += F.nll_loss(y_hat, y, reduction='sum').item()
            pred = y_hat.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(y.data.view_as(pred)).sum()
        test_loss /= n_total
        print(f'テストデータでの平均損失 {test_loss}')
        print('テストデータでの正解率 {}/{} ({:.2%})'.format(int(n_correct), n_total, n_correct / n_total))


# メイン
#  - arch : 学習するモデル構造を gru, tcn から指定します
#  - id : 吐き出す重みファイルの識別子です
#  - weight_dict : 既に重みファイルがあれば読み込みます
#  - epochs : エポック数です 0にすると訓練スキップになります
#  - permute : これを指定すると系列をこのインデックスの順序に入れ換えます
def main(arch='gru', id='hoge', weight_dict=None, epochs=10, permute=None):
    batch_size = 64 
    train_loader, test_loader = MNIST(batch_size=batch_size,
                                      sequential=(arch == 'tcn'),
                                      sequential_rnn=(arch != 'tcn'),
                                      permute=permute)
    if arch == 'tcn':
        model = TCN(input_size=1, output_size=10, num_channels=[25]*8,
                    kernel_size=7, dropout=0.0)
        optimizer = optim.Adam(model.parameters(), lr=2e-3)
    elif arch == 'gru':
        model = GRU(input_size=1, output_size=10, num_layers=1, d_hidden=128,
                    initial_update_gate_bias=0.5, dropout=0.0)
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3) 

    if weight_dict is not None:
        model.load_state_dict(torch.load(weight_dict))

    for epoch in range(epochs):
        print(f'エポック {epoch}')
        train(model, optimizer, train_loader)
        test(model, test_loader)
        torch.save(model.state_dict(), f'./weights/{arch}_sequential_mnist_{id}_{epoch}.dict')

    test(model, test_loader)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    main(arch='gru', weight_dict='./weights/gru_sequential_mnist_sample.dict', epochs=0)
    main(arch='tcn', weight_dict='./weights/tcn_sequential_mnist_sample.dict', epochs=0)
    # main(arch='gru', epochs=1)
    # main(arch='tcn', epochs=1)

    # Permuted MNIST をする場合
    permute = np.random.permutation(784)
    # main(arch='gru', epochs=1, permute=permute)
    # main(arch='tcn', epochs=1, permute=permute)
