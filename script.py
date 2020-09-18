import torch
from transformers import (
    # BertConfig,
    BertTokenizer,
    BertForTokenClassification,
)
from utils import count_parameters


def main():
    # 各トークンを以下の13クラスのいずれかに分類するような固有表現抽出をしたい．
    labels = [
        'B-corporation',
        'B-creative-work',
        'B-group',
        'B-location',
        'B-person',
        'B-product',
        'I-corporation',
        'I-creative-work',
        'I-group',
        'I-location',
        'I-person',
        'I-product',
        'O'
    ]
    id2label = {i: label for i, label in enumerate(labels)}
    # label2id = {label: i for i, label in enumerate(labels)}

    # 利用する学習済みBERTモデルの名前を指定する．
    model_name = 'bert-large-cased'

    # 学習済みモデルに対応したトークナイザを生成する．
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
    )

    # 学習済みモデルから各トークン分類用モデルのインスタンスを生成する．
    model = BertForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        id2label=id2label,  # 各トークンに対する出力を13次元にしたいのでこれを渡す．
    )
    # 一部の重みが初期化されていませんよという警告が出るが（クラス分類する層が
    # 初期化されていないのは当然）面倒なので無視する．

    n = 0
    blocks = [
        ('モデルの埋め込み層', model.bert.embeddings),
        ('モデルのエンコーダ層', model.bert.encoder),
        ('モデルのプーラー層', model.bert.pooler),
        # ('モデルの全結合層', model.classifier),
    ]
    for block in blocks:
        print('-'*10, block[0], '-'*10)
        if block[0] == 'モデルのエンコーダ層':
            for i, layer in enumerate(block[1].layer):
                n_layer = 0
                for name, param in layer.named_parameters():
                    n_ = param.numel()
                    if i == 0:
                        print(name, n_)
                    n_layer += n_
                print(f'エンコーダ層内の{i}層目計', n_layer)
                n += n_layer
        else:
            for name, param in block[1].named_parameters():
                n_ = param.numel()
                print(name, n_)
                n += n_

    print('='*10, 'パラメータ数', '='*10)
    print(n)

    # print('◆ 適当な文章をモデルに流してみる．→ 14トークン×13クラスの予測結果になっている（サイズが）．')
    # sentence = 'The Empire State Building officially opened on May 1, 1931.'
    # inputs = torch.tensor([tokenizer.encode(sentence)])  # ID列をテンソル化して渡す．
    # outputs = model(inputs)
    # print(outputs[0].size())


if __name__ == '__main__':
    main()
