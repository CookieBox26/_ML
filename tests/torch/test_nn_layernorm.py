import torch
from torch import nn
import math


class TestLayerNorm:

    def test(self):
        """ LayerNorm を検算する．
        """
        # 適当なテンソルをつくる．
        # 2つの3語文（各単語は4次元に埋め込まれている）が入ったバッチのイメージ．
        x = torch.tensor([[[1., 2., 3., 4.],      # 1文目の1単語目
                           [5., 6., 7., 8.],      # 1文目の2単語目
                           [9., 10., 11., 12.]],  # 1文目の3単語目
                          [[1., 2., 1., 2.],      # 2文目の1単語目
                           [-1., -2., -1., -2.],  # 2文目の2単語目
                           [1., 2., 1., 2.]]])    # 2文目の3単語目
        assert x.size() == torch.Size([2, 3, 4])

        # ---------- 例1. 単語レベルの正規化 ----------

        # 単語レベルで正規化するとしたらこう．
        model = nn.LayerNorm(normalized_shape=4, eps=1e-12, elementwise_affine=True)

        # 含まれるパラメータは weight と bias である（アフィン変換を有効にするとこれを学習）．
        # それぞれ4次元である．
        assert type(model.weight) is torch.nn.parameter.Parameter
        assert model.weight.size() == torch.Size([4])
        assert model.weight.requires_grad
        assert type(model.bias) is torch.nn.parameter.Parameter
        assert model.bias.size() == torch.Size([4])
        assert model.bias.requires_grad

        # パラメータのデフォルト値はこう．
        assert torch.equal(model.weight.data, torch.tensor([1., 1., 1., 1.]))
        assert torch.equal(model.bias.data, torch.tensor([0., 0., 0., 0.]))

        # さっきつくった適当なテンソルを LayerNorm する．
        y = model(x)
        assert y.size() == torch.Size([2, 3, 4])
        # tensor([[[-1.3416, -0.4472,  0.4472,  1.3416],
        #          [-1.3416, -0.4472,  0.4472,  1.3416],
        #          [-1.3416, -0.4472,  0.4472,  1.3416]],
        #         [[-1.0000,  1.0000, -1.0000,  1.0000],
        #          [ 1.0000, -1.0000,  1.0000, -1.0000],
        #          [-1.0000,  1.0000, -1.0000,  1.0000]]],
        #        grad_fn=<NativeLayerNormBackward>)

        # 値を検算する．
        # 各単語ベクトルが、平均をマイナスして分散の平方根で割ったものになっていればよい．
        # ※ 1をかけて0を足しても何も変わらないので weight と bias の反映は省略．
        for i_sentence in range(2):
            for i_word in range(3):
                # i_sentence 文目の i_word 語目の単語の4次元ベクトル
                word = x[i_sentence, i_word, :]
                mean = torch.mean(word)  # 平均
                var = torch.var(word, unbiased=False)  # 分散  ※ 不偏分散にしないこと！
                word_mod_expected = (word - mean) / torch.sqrt(var + 1e-12)  # LayerNorm
                word_mod_actual = y[i_sentence, i_word, :]
                assert word_mod_actual.size() == word_mod_expected.size()
                assert all([math.isclose(el_actual.item(), el_expected.item(), abs_tol=1e-6)
                            for (el_actual, el_expected)
                            in zip(word_mod_actual, word_mod_expected)])

        # ---------- 例2. 文章レベルの正規化 ----------

        # 文章レベルで正規化するとしたらこう.
        model = nn.LayerNorm(normalized_shape=(3, 4), eps=1e-12, elementwise_affine=True)
        y = model(x)
        # tensor([[[-1.5933, -1.3036, -1.0139, -0.7242],
        #          [-0.4345, -0.1448,  0.1448,  0.4345],
        #          [ 0.7242,  1.0139,  1.3036,  1.5933]],
        #         [[ 0.3333,  1.0000,  0.3333,  1.0000],
        #          [-1.0000, -1.6667, -1.0000, -1.6667],
        #          [ 0.3333,  1.0000,  0.3333,  1.0000]]],
        #        grad_fn=<NativeLayerNormBackward>)

        # 値を検算する．
        # 各文章が、平均をマイナスして分散の平方根で割ったものになっていればよい．
        # ※ 1をかけて0を足しても何も変わらないので weight と bias の反映は省略．
        for i_sentence in range(2):
            # i_sentence 文目
            sentence = x[i_sentence, :, :]
            mean = torch.mean(sentence)  # 平均
            var = torch.var(sentence, unbiased=False)  # 分散  ※ 不偏分散にしないこと！
            sentence_mod_expected = (sentence - mean) / torch.sqrt(var + 1e-12)  # LayerNorm
            sentence_mod_actual = y[i_sentence, :, :]
            assert sentence_mod_actual.size() == sentence_mod_expected.size()
            assert all([math.isclose(el_actual.item(), el_expected.item(), abs_tol=1e-6)
                        for (el_actual, el_expected)
                        in zip(sentence_mod_actual.view(-1), sentence_mod_expected.view(-1))])

        # ---------- 例3. バッチレベルの正規化 ----------

        # バッチレベルで正規化するとしたらこう．
        model = nn.LayerNorm(normalized_shape=(2, 3, 4), eps=1e-12, elementwise_affine=True)
        y = model(x)
        # tensor([[[-0.6234, -0.3740, -0.1247,  0.1247],
        #          [ 0.3740,  0.6234,  0.8727,  1.1221],
        #          [ 1.3714,  1.6208,  1.8701,  2.1195]],
        #         [[-0.6234, -0.3740, -0.6234, -0.3740],
        #          [-1.1221, -1.3714, -1.1221, -1.3714],
        #          [-0.6234, -0.3740, -0.6234, -0.3740]]],
        #        grad_fn=<NativeLayerNormBackward>)

        # 値を検算する．
        mean = torch.mean(x)  # 平均
        var = torch.var(x, unbiased=False)  # 分散  ※ 不偏分散にしないこと！
        x_mod_expected = (x - mean) / torch.sqrt(var + 1e-12)  # LayerNorm
        assert y.size() == x_mod_expected.size()
        assert all([math.isclose(el_actual.item(), el_expected.item(), abs_tol=1e-6)
                    for (el_actual, el_expected)
                    in zip(y.view(-1), x_mod_expected.view(-1))])
