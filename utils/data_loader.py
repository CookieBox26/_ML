import torch
from torchvision import datasets, transforms


# MNIST をロードします
#  - for i_batch, (x, y) in enumerate(train_loader) のように使います
#  - x のサイズが [batch_size, 1, 28, 28], y のサイズが [batch_size] になります
#  - sequential=True で x のサイズが [batch_size, 1, 784] になります
#  - sequential_rnn=True で x のサイズが [batch_size, 784, 1] になります
#  - (0.1307,), (0.3081,) は MNIST の訓練データの平均と標準偏差らしいです
def MNIST(root='./data/', batch_size=64, sequential=False, sequential_rnn=False):
    my_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if sequential or sequential_rnn:
        my_transforms.append(transforms.Lambda(lambda x: x.view(-1, 784)))
    if sequential_rnn:
        my_transforms.append(transforms.Lambda(lambda x: x.transpose(0, 1)))
    my_transforms = transforms.Compose(my_transforms)
    train_set = datasets.MNIST(root=root, train=True, download=True, transform=my_transforms)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=my_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader
