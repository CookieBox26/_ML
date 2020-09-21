# TrellisNet で文字レベル Penn Treebank を学習する．

import torch
from torch import nn
import torch.optim as optim
from observations import ptb
from trellisnet.model import TrellisNetModel


def batchify(data, batch_size):
    """
    バッチサイズが4の場合：
    元の系列がこうなっているとする．
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    モデルは4本ずつ食えるので4分割する（端数は捨てる）．
    tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]])
    同一ステップで食われるべき要素がベクトルになるように転置する．
    tensor([[0, 3, 6, 9],   <-- 0ステップ目
            [1, 4, 7, 10],  <-- 1ステップ目
            [2, 5, 8, 11]]) <-- 2ステップ目
    """
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)  # 端数は捨てる．
    return data.view(batch_size, -1).t().contiguous()  # 転置する．


def main():
    # Penn Treebank を取得する．
    # 訓練用，テスト用，検証用のコーパス（文字列）が取得される．
    x_train, x_test, x_valid = ptb("./data")  # 初回はダウンロードが走る．

    # コーパスに含まれる <eos> を1文字扱いにする．未使用の chr(255) を割り当てる．
    # メモ： <unk> は1文字扱いされていないがいいのだろうか．
    x_train = x_train.replace('<eos>', chr(255))
    x_valid = x_valid.replace('<eos>', chr(255))
    x_test = x_test.replace('<eos>', chr(255))
    print(f'訓練データ： {len(x_train)} 字')
    print(f'検証データ： {len(x_valid)} 字')
    print(f'テストデータ： {len(x_test)} 字')

    # コーパスに登場する文字にインデックスをふる．
    set_char = set([c for c in x_train]) | set([c for c in x_test]) | set([c for c in x_valid])
    idx2char = list(set_char)
    idx2char = sorted(idx2char, key=lambda c: ord(c))  # 何となくアスキーコード順でソート．
    char2idx = {c: i for i, c in enumerate(idx2char)}
    ntoken = len(idx2char)  # 文字レベルの系列にしているのでトークンというより文字だが．
    print(f'ユニーク文字数： {ntoken} 字')
    print(char2idx)

    # データをIDのテンソル化しておく．
    x_train = torch.tensor([char2idx[c] for c in x_train])
    x_valid = torch.tensor([char2idx[c] for c in x_valid])
    x_test = torch.tensor([char2idx[c] for c in x_test])

    # 訓練データをバッチサイズ本ずつモデルに食わせられるように整形する．
    # torch.Size([5101618]) --> torch.Size([212567, 24])
    batch_size = 24
    x_train = batchify(x_train, batch_size)

    # モデル，損失関数，オプティマイザを用意する．
    model = TrellisNetModel(ntoken=ntoken, ninp=200, nhid=1050, nout=200, nlevels=140)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=8e-7)

    # 3バッチ（1バッチ = バッチサイズ24 × シーケンス長40）だけ訓練する．
    model.train()
    aux = 0.3
    clip = 0.2
    seq_len = 40
    for batch, i in enumerate(range(0, x_train.size(0) - 1, seq_len)):
        optimizer.zero_grad()
        data = x_train[i:(i + seq_len)].t()

        hidden = model.init_hidden(batch_size)
        net = nn.DataParallel(model)
        (_, _, decoded), hidden, all_decoded = net(data, hidden)
        decoded = decoded.transpose(0, 1)

        targets = x_train[(i + 1):(i + 1 + seq_len)].contiguous().view(-1)
        final_decoded = decoded.contiguous().view(-1, ntoken)
        raw_loss = criterion(final_decoded, targets)

        all_decoded = all_decoded.permute(1, 2, 0, 3).contiguous()
        aux_size = all_decoded.size(0)
        all_decoded = all_decoded.view(aux_size, -1, ntoken)
        aux_losses = aux * sum([criterion(all_decoded[i], targets) for i in range(aux_size)])

        loss = raw_loss + aux_losses
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        print(f'{batch} バッチ目', raw_loss)
        if batch == 2:
            break


if __name__ == '__main__':
    main()
