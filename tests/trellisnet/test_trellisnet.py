import torch
from torch import nn
from trellisnet.optimizations import WeightNorm
from trellisnet.trellisnet import TrellisNet
from trellisnet.model import TrellisNetModel


class TestWeightNorm:

    def test(self):
        m = nn.Linear(3, 4)
        assert m.weight.size() == torch.Size([4, 3])
        assert m.bias.size() == torch.Size([4])

        fn = WeightNorm.apply(m, ['weight'], 0)
        assert type(fn) is WeightNorm
        assert m.weight_g.size() == torch.Size([4, 1])
        assert m.weight_v.size() == torch.Size([4, 3])
        assert m.bias.size() == torch.Size([4])


class TestTrellisNet:

    def test(self):
        model = TrellisNet(ninp=200, nhid=1050, nout=200)
        assert type(model) is TrellisNet


class TestTrellisNetModel:

    def test(self):
        """
        """
        model = TrellisNetModel(ntoken=50, ninp=200, nhid=1050, nout=200, nlevels=140)

        batch_size = 4
        inputs = torch.tensor([
            [0, 23, 27, 40, 0, 24, 23, 36, 33, 36, 37, 42, 27, 0, 24, 27],
            [27, 41, 0, 23, 34, 42, 30, 37, 43, 29, 30, 0, 30, 27, 0, 45],
            [34, 37, 41, 41, 0, 26, 31, 41, 25, 34, 37, 41, 27, 26, 0, 31],
            [31, 41, 41, 43, 27, 41, 0, 31, 36, 0, 45, 30, 31, 25, 30, 0]]
        )

        hidden = model.init_hidden(batch_size)
        assert hidden[0].size() == torch.Size([4, 1250, 1])
        assert hidden[1].size() == torch.Size([4, 1250, 1])

        (_, _, decoded), hidden, all_decoded = model(inputs, hidden)
        assert decoded.size() == torch.Size([4, 16, 50])
        assert hidden[0].size() == torch.Size([4, 1250, 1])
        assert hidden[1].size() == torch.Size([4, 1250, 1])
        assert all_decoded.size() == torch.Size([4, 7, 16, 50])
