# ML

機械学習モデルを動かしてみるためのリポジトリです。PyTorch を利用しています。その他、参考リンクの機械学習モデルをインポートしています。

### Quick Usage

#### pipenv を利用しない場合
スクリプトを実行する ```python script_ner_with_bert.py```  
テストを実行する ```pytest --ignore=./tests/trellisnet --ignore=./TCN/```

#### pipenv を利用する場合
環境をインストールする ```pipenv install --dev```  
スクリプトを実行する ```pipenv run python script_ner_with_bert.py```  
テストを実行する ```pipenv run pytest --ignore=./tests/trellisnet --ignore=./TCN/```

<h5>Windows で pipenv を利用する場合</h5>

- 予め https://www.python.org/downloads/ から Python3.7 をインストールして python と pip にパスを通し、```pip install pipenv``` で pipenv をインストールしてください。
- Pipfile 内にある通り ```https://download.pytorch.org/whl/cu102/torch-1.6.0-cp37-cp37m-win_amd64.whl``` から PyTorch をインストールします。GPU 環境でない場合や CUDA 10.2 でない場合などは https://download.pytorch.org/whl/torch_stable.html からお手元のマシンにインストールできる wheel ファイルを探して書き換えてください。
- <b>Pipfile からインストールする前に ```torch = {version = "==1.6.0", sys_platform = "!= 'win32'"}``` の行を明示的にコメントアウトしてください（重要）。</b>
- 後は通常通り ```pipenv install --dev``` で環境をインストールしてください。Python が見つからずに失敗する場合は ```pipenv install --python 3.7 --dev``` としてみてください。

### 内容物の説明
- ./script_xxx.py
    - 各種の機械学習タスクを実行するスクリプトです。が、まだ工事中です。
    - TrellisNet を利用するスクリプトの場合は、./trellisnet/ を取得してから実行する必要があります。
- ./tests/
    - テストですが、テストとみせかけて、各種モデルの仕様のメモです。```pytest``` でテストを実行します。ただし、
        - ./trellisnet/ を取得していない場合は TrellisNet のテストはできないので ```pytest --ignore=./tests/trellisnet``` とする必要があります。
        - ./TCN/ を取得している場合、このリポジトリ内にファイル名に test が付くファイルがあるので ```--ignore=./TCN/``` も付ける必要があります。
    - 環境変数 ```export SKIP_BERT=TRUE``` を設定すると BERT を読み込むテスト（時間がかかる）をスキップできます。
- ./data/
    - 取得したデータを置くところです。
- ./models/
    - 自分で定義したモデルを置くところです。
- ./weights/
    - 学習したモデルの重みを置くところです。
- ./utils/
    - 自分で定義した便利関数を置くところです。
- ./TCN/
    - TCN のソースコードを置くところです。デフォルトで同梱していません。このディレクトリ内で以下のように取得してください。なお、TCN の本家のリポジトリのフォルダ構成ではモデルを読み込めないので、必ず CookieBox26 の Fork を取得してください。<br/> ```git clone https://github.com/CookieBox26/TCN.git```
- ./trellisnet/
    - TrellisNet のソースコードを置くところです。デフォルトで同梱していません。このディレクトリ内で以下のように取得してください。なお、TrellisNet の本家のリポジトリのフォルダ構成ではモデルを読み込めないので、必ず CookieBox26 の Fork を取得してください。<br/> ```git clone https://github.com/CookieBox26/trellisnet.git```

### 各スクリプトの説明

#### script_sequential_mnist.py
MNISTを1次元系列として扱ってどの数字が分類するタスクをGRUまたはTCNで解きます。GRUについては10エポック学習したパラメータを同梱しています。これを指定して訓練をスキップすると以下のように表示されます。
```
テストデータでの平均損失 0.09641245974451304
テストデータでの正解率 9694/10000 (96.94%)
```

TCNはまだ学習済みパラメータを同梱していません。  
どちらのモデルでも正のエポック数を指定すると以下のように学習が始まります。
```
エポック 0
10/938 バッチ (640/60000 サンプル) 流れました  最近 10 バッチの平均損失 2.3024290084838865
20/938 バッチ (1280/60000 サンプル) 流れました  最近 10 バッチの平均損失 2.2999905586242675
30/938 バッチ (1920/60000 サンプル) 流れました  最近 10 バッチの平均損失 2.277213621139526
40/938 バッチ (2560/60000 サンプル) 流れました  最近 10 バッチの平均損失 1.9509142518043519
50/938 バッチ (3200/60000 サンプル) 流れました  最近 10 バッチの平均損失 1.236298155784607
```

以下補足です。
- 手抜きのためにTCNの著者のコードではなく自分でクラスをかきかえたコードを参照していますが動作は同じです。
- TCNのネットワーク構造とオプティマイザは[TCNの原論文](https://arxiv.org/abs/1803.01271)に倣っています。
- GRUのネットワーク構造とオプティマイザは同論文のLSTMのセッティングに似せたものです（Grad Clip はしていません）。
- バッチサイズ64は著者の TCN による Seq. MNIST のコードにあったデフォルト値です。
    - https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/pmnist_test.py#L13-L14

#### script_ner_with_bert.py
WNUT’17 の固有表現抽出タスクをしようとしていますが、まだ適当な文章をモデルに流すところまでしか実装されていません。以下が標準出力に出力されるだけです。
```
# ここに Some weights of the model checkpoint at bert-large-cased were not used... に始まる警告文が出る．
◆ 適当な文章をモデルに流してみる．→ 14トークン×13クラスの予測結果になっている（サイズが）．
torch.Size([1, 14, 13])
```

#### script_char_ptb_with_trellisnet.py
※ このスクリプトを実行するには ./trellisnet/ の取得が必要です。  
文字レベルの Penn Treebank 予測タスクをしようとしていますが、まだ以下が標準出力に出力されるだけです。
```
訓練データ： 5101618 字
検証データ： 399782 字
テストデータ： 449945 字
ユニーク文字数： 50 字
{' ': 0, '#': 1, '$': 2, '&': 3, "'": 4, '*': 5, '-': 6, '.': 7, '/': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, '<': 19, '>': 20, 'N': 21, '\\': 22, 'a': 23, 'b': 24, 'c': 25, 'd': 26, 'e': 27, 'f': 28, 'g': 29, 'h': 30, 'i': 31, 'j': 32, 'k': 33, 'l': 34, 'm': 35, 'n': 36, 'o': 37, 'p': 38, 'q': 39, 'r': 40, 's': 41, 't': 42, 'u': 43, 'v': 44, 'w': 45, 'x': 46, 'y': 47, 'z': 48, 'ÿ': 49}
Weight normalization applied
0 バッチ目 tensor(3.9125, grad_fn=<NllLossBackward>)
1 バッチ目 tensor(3.8557, grad_fn=<NllLossBackward>)
2 バッチ目 tensor(3.6744, grad_fn=<NllLossBackward>)
```
```--use_cuda``` を付けて実行すると GPU で学習します。
```
0 バッチ目 tensor(3.9117, device='cuda:0', grad_fn=<NllLossBackward>)
1 バッチ目 tensor(3.8551, device='cuda:0', grad_fn=<NllLossBackward>)
2 バッチ目 tensor(3.4490, device='cuda:0', grad_fn=<NllLossBackward>)
```

### 参考リンク
- https://github.com/pytorch/pytorch/tree/v1.6.0
    - PyTorch のリポジトリです（v1.6.0）。
    - 特に nn.Module のソースは以下です。
        - https://github.com/pytorch/pytorch/blob/v1.6.0/torch/nn/modules/module.py
- https://github.com/huggingface/transformers/tree/v3.1.0
    - transformers のリポジトリです（v3.1.0）。
    - 特に BERT モデルのソースは以下です。
        - https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/modeling_bert.py
- https://github.com/locuslab/TCN
    - [TCN](https://arxiv.org/abs/1803.01271) のリポジトリです。
    - 使い勝手のために以下にフォークしています。
        - https://github.com/CookieBox26/TCN
- https://github.com/locuslab/trellisnet
    - [TrellisNet](https://arxiv.org/abs/1810.06682) のリポジトリです。
    - 使い勝手のために以下にフォークしています。
        - https://github.com/CookieBox26/trellisnet
