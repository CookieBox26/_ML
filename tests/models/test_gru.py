import torch
from models.gru import GRU
from pytest import approx


def count_parameters(model):
    """ パラメータ数を取得する．
    """
    n = 0
    for param in model.parameters():
        if param.requires_grad:
            n += param.numel()
    return n


class TestGRU:

    # ラップした GRU クラス
    def test(self):
        model = GRU(input_size=1,
                    output_size=8,
                    num_layers=2,
                    d_hidden=64)

        assert list(model.gru.weight_ih_l0.size()) == [192, 1]
        assert list(model.gru.weight_hh_l0.size()) == [192, 64]
        assert list(model.gru.bias_ih_l0.size()) == [192]
        assert list(model.gru.bias_hh_l0.size()) == [192]
        assert list(model.gru.weight_ih_l1.size()) == [192, 64]
        assert list(model.gru.weight_hh_l1.size()) == [192, 64]
        assert list(model.gru.bias_ih_l1.size()) == [192]
        assert list(model.gru.bias_hh_l1.size()) == [192]
        assert list(model.linear.weight.size()) == [8, 64]
        assert list(model.linear.bias.size()) == [8]

        n = 192
        n += 192 * 64
        n += 192
        n += 192
        n += 192 * 64
        n += 192 * 64
        n += 192
        n += 192
        n += 8 * 64
        n += 8
        assert count_parameters(model) == n

    def test_load_weight(self):
        # state_dict の実体は OrderedDict
        # キーがパラメータ名で値が PyTorch のテンソル
        state_dict = torch.load('./weights/gru_sequential_mnist_sample.dict')

        # 含まれているネットワークの重み
        weights_expected = [
            ('gru.weight_ih_l0', [384, 1]),
            ('gru.weight_hh_l0', [384, 128]),
            ('gru.bias_ih_l0', [384]),
            ('gru.bias_hh_l0', [384]),
            ('linear.weight', [10, 128]),
            ('linear.bias', [10]),
        ]

        for i, (key, value) in enumerate(state_dict.items()):
            assert key == weights_expected[i][0]
            assert list(value.size()) == weights_expected[i][1]

        # 整合性が合うモデルにならロードできる
        torch.manual_seed(0)
        model = GRU(input_size=1,    # Sequential MNIST タスクなら入力は1次元
                    output_size=10,  # Sequential MNIST タスクなら出力は10次元
                    num_layers=1,    # GRU ブロックの積み重ね数
                    d_hidden=128)    # GRU ブロックの出力次元数（隠れ状態の次元数）

        # ロード前後で最初の重みが変わることを確認
        assert model.gru.weight_ih_l0.data[0, 0].item() == approx(-0.0006617, rel=1e-3)
        model.load_state_dict(state_dict)
        assert model.gru.weight_ih_l0.data[0, 0].item() == approx(0.2248, rel=1e-3)
