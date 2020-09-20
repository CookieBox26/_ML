from models.gru import GRU
from utils import count_parameters


class TestGRU:

    # 自分でラップした GRU クラス
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
