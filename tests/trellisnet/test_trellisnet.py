from trellisnet.model import TrellisNetModel
from trellisnet.trellisnet import TrellisNet


class TestTrellisNet:

    def test(self):
        model = TrellisNet(ninp=200, nhid=1050, nout=200)


class TestTrellisNetModel:

    def test(self):
        model = TrellisNetModel(ntoken=50, ninp=200, nhid=1050, nout=200, nlevels=140)
