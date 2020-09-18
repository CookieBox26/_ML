from torch import nn
import types


class TestModule:

    def test(self):
        model = nn.GRU(input_size=1,
                       hidden_size=8,
                       num_layers=2)

        # parameters() の返り値はジェネレータオブジェクトである
        params = model.parameters()
        assert isinstance(params, types.GeneratorType)
        # print(params[0])  # なのでこういうことはできない

        # named_parameters() の返り値はジェネレータオブジェクトである
        params = model.named_parameters()
        assert isinstance(params, types.GeneratorType)
