import torch
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
)

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
    label2id = {label: i for i, label in enumerate(labels)}

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
    # print(model)  # 24層あるのでプリントすると長い．


    print('◆ 適当な文章をID列にしてみる．')
    sentence = 'The Empire State Building officially opened on May 1, 1931.'

    # BERT に文章を流すとき文頭に特殊トークン [CLS] 、
    # 文末に特殊トークン [SEP] が想定されている．
    # tokenizer.encode() でID列にすると勝手に付加されている．
    print('◇')
    ids = tokenizer.encode(sentence)
    for id_ in ids:
        token = tokenizer.convert_ids_to_tokens(id_)
        print(str(id_).ljust(5), tokenizer.convert_ids_to_tokens(id_))

    # 先にトークン列が手元にある場合は特殊トークンを明示的に付加する．
    print('◇')
    tokens = tokenizer.tokenize(sentence)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    for token in tokens:
        id_ = tokenizer.convert_tokens_to_ids(token)
        print(str(id_).ljust(5), tokenizer.convert_ids_to_tokens(id_))

    print('◆ モデルに流してみる．→ 14トークン×13クラスの予測結果になっている（サイズが）．')
    inputs = torch.tensor([tokenizer.encode(sentence)])  # ID列をテンソル化して渡す．
    outputs = model(inputs)
    print(outputs[0].size())


if __name__ == '__main__':
    main()
