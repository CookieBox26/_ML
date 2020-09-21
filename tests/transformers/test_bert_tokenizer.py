import os
import pytest
from transformers import BertTokenizer


@pytest.fixture(scope='class')
def tokenizer():
    return BertTokenizer.from_pretrained(
               pretrained_model_name_or_path='bert-large-cased',
           )


@pytest.mark.skipif(os.environ.get('SKIP_BERT', '') == 'TRUE', reason='take time')
class TestBertTokenizer:

    def test(self, tokenizer):
        # BERT に文章を流すとき文頭に特殊トークン [CLS] 、
        # 文末に特殊トークン [SEP] が想定されている．
        n = tokenizer.num_special_tokens_to_add()
        assert n == 2  # [CLS], [SEP]
        assert tokenizer.cls_token == '[CLS]'
        assert tokenizer.sep_token == '[SEP]'
        assert tokenizer.cls_token_id == 101
        assert tokenizer.sep_token_id == 102

        # 適当な文章をID列にしてみる．
        sentence = 'The Empire State Building officially opened on May 1, 1931.'
        expected = [
            (101, '[CLS]'),
            (1109, 'The'),
            (2813, 'Empire'),
            (1426, 'State'),
            (4334, 'Building'),
            (3184, 'officially'),
            (1533, 'opened'),
            (1113, 'on'),
            (1318, 'May'),
            (122, '1'),
            (117, ','),
            (3916, '1931'),
            (119, '.'),
            (102, '[SEP]')
        ]

        # encode() でID列にすると勝手に [CLS], [SEP] が付加されている．
        ids = tokenizer.encode(sentence)
        assert len(ids) == len(expected)
        for id_, (id_expected, token_expected) in zip(ids, expected):
            token = tokenizer.convert_ids_to_tokens(id_)  # id --> token
            assert id_ == id_expected
            assert token == token_expected

        # tokenize() でトークナイズすると勝手に付加されていない．
        tokens = tokenizer.tokenize(sentence)
        assert len(tokens) == len(expected) - 2
        for token, (id_expected, token_expected) in zip(tokens, expected[1:-1]):
            id_ = tokenizer.convert_tokens_to_ids(token)  # token --> id
            assert id_ == id_expected
            assert token == token_expected
