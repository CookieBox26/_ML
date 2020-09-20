# ML

機械学習モデルを動かしてみるためのリポジトリです。PyTorch 1.4.0 を利用しています。その他、参考リンクの機械学習モデルをインポートしています。

### Quick Usage

#### pipenv を利用しない場合
スクリプトを実行する ```python script_ner_with_bert.py```  
テストを実行する ```pytest --ignore=tests/trellisnet```

#### pipenv を利用する場合
環境をインストールする ```pipenv install --dev```  
スクリプトを実行する ```pipenv run python script_ner_with_bert.py```  
テストを実行する ```pipenv run pytest --ignore=tests/trellisnet```

### 内容物の説明
- ./script_xxx.py
    - 各種の機械学習タスクを実行するスクリプトです。が、まだ工事中です。
    - TrellisNet を利用するスクリプトの場合は、./trellisnet/ を取得してから実行する必要があります。
- ./tests/
    - テストですが、テストとみせかけて、各種モデルの仕様のメモです。```pytest``` ですべてのテストを実行しますが、./trellisnet/ を取得していない場合は TrellisNet のテストはできないので ```pytest --ignore=tests/trellisnet``` とする必要があります。
    - 環境変数 ```export SKIP_BERT=TRUE``` を設定すると BERT を読み込むテスト（時間がかかる）をスキップできます。
- ./models/
    - 自分で定義したモデルを置くところです。
- ./utils.py
    - 自分で定義した便利関数を書くところです。
- ./trellisnet/
    - TrellisNet のソースコードを置くところです。デフォルトで同梱していません。このディレクトリ内で以下のように取得してください。なお、TrellisNet の本家のリポジトリのフォルダ構成ではモデルを読み込めないので、必ず CookieBox26 の Fork を取得してください。<br/> ```git clone https://github.com/CookieBox26/trellisnet.git```


### 各スクリプトの説明

#### script_ner_with_bert.py
WNUT’17 の固有表現抽出タスクをしようとしていますが、まだ適当な文章をモデルに流すところまでしか実装されていません。以下が標準出力に出力されるだけです。
```
# ここに Some weights of the model checkpoint at bert-large-cased were not used... に始まる警告文が出る．
◆ 適当な文章をモデルに流してみる．→ 14トークン×13クラスの予測結果になっている（サイズが）．
torch.Size([1, 14, 13])
```

#### script_char_ptb_with_trellisnet.py
※ このスクリプトを実行するには ./trellisnet/ の取得が必要です。  
文字レベルの Penn Treebank 予測タスクをしようとしていますが、モデルインスタンスの作成しかしていません。

### 参考リンク
- https://github.com/pytorch/pytorch/tree/v1.4.0
    - PyTorch のリポジトリです（v1.4.0）。
    - 特に nn.Module のソースは以下です。
        - https://github.com/pytorch/pytorch/blob/v1.4.0/torch/nn/modules/module.py
- https://github.com/huggingface/transformers/tree/v3.1.0
    - transformers のリポジトリです（v3.1.0）。
    - 特に BERT モデルのソースは以下です。
        - https://github.com/huggingface/transformers/blob/v3.1.0/src/transformers/modeling_bert.py
- https://github.com/locuslab/trellisnet
    - TrellisNet のリポジトリです。
    - 使い勝手のために以下にフォークしています。
        - https://github.com/CookieBox26/trellisnet
