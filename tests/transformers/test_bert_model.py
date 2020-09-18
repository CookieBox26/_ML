import pytest
# import torch
from torch import nn
from transformers import (
    BertConfig,
    # BertTokenizer,
    # BertEmbeddings,  # このクラスは公開されていない．
    # BertEncoder,  # このクラスは公開されていない．
    # BertPooler,  # このクラスは公開されていない．
    BertModel,
    BertForTokenClassification,
)


@pytest.fixture(scope='class')
def model():
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
    return BertForTokenClassification.from_pretrained(
               pretrained_model_name_or_path='bert-large-cased',
               id2label=id2label,  # 各トークンに対する出力を13次元にしたいのでこれを渡す．
           )


class TestBertModel:

    def test(self, model):
        # モデルのコンフィグレーション（BertConfigクラス）
        assert type(model.config) is BertConfig
        assert model.config.hidden_size == 1024

        # モデルのうち学習済みBERTモデル部分
        assert type(model.bert) is BertModel
        # モデルのうち固有表現抽出のために付け足した部分
        assert type(model.dropout) is nn.Dropout
        assert type(model.classifier) is nn.Linear

        # 学習済みBERTモデルの詳細（埋め込み層、エンコーダ層、プーラー層）
        # assert type(model.bert.embeddings) is BertEmbeddings
        # assert type(model.bert.encoder) is BertEncoder
        # assert type(model.bert.pooler) is BertPooler
        assert isinstance(model.bert.embeddings, nn.Module)
        assert isinstance(model.bert.encoder, nn.Module)
        assert isinstance(model.bert.pooler, nn.Module)

        # 埋め込み層の詳細
        assert type(model.bert.embeddings.word_embeddings) is nn.Embedding
        assert type(model.bert.embeddings.position_embeddings) is nn.Embedding
        assert type(model.bert.embeddings.token_type_embeddings) is nn.Embedding
        assert type(model.bert.embeddings.LayerNorm) is nn.LayerNorm
        assert type(model.bert.embeddings.dropout) is nn.Dropout
