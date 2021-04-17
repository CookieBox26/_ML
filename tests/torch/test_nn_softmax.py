import torch
from torch import nn
from pytest import approx


class TestSoftmax:

    def test(self):
        # ソフトマックスの方向がよくわからなくなった
        # ソフトマックスする直前の状態が以下とする
        # 4次元配列であってサイズが [バッチサイズ, ヘッド数, 文章長さ, 文章長さ] である
        attention_scores = torch.Tensor(
            [[[
                [1.0, 2.0, 1.0],
                [1.0, 3.0, 1.5],
                [1.0, 5.0, 1.0],
            ]]]
        )
        assert list(attention_scores.size()) == [1, 1, 3, 3]
        # それでこんな風にソフトマックスしている
        # 4次元配列の3次元目をソフトマックスしている
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        print(attention_probs)
        # そうすると4次元配列の3次元目の方向の和が1になる
        s = torch.sum(attention_probs, 3)
        assert s[0, 0, 0].item() == approx(1.0, rel=1e-6)
        # 4次元配列の2次元目の方向の和は1にならない
        s = torch.sum(attention_probs, 2)
        assert s[0, 0, 0].item() != approx(1.0, rel=1e-6)
        # 結局一番内側にみえているリストが正規化される

        # そもそも2次元配列の行と列がよくわからないが
        # 以下のようにかけば普通に視覚的に行列
        # つまり一番内側のリストとしてみえているのは行
        a = torch.Tensor([[1, 1, 1],
                          [1, 1, 1]])
        b = torch.Tensor([[1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1]])
        c = torch.matmul(a, b)
        assert list(a.size()) == [2, 3]
        assert list(b.size()) == [3, 4]
        assert list(c.size()) == [2, 4]

        # a は2行3列
