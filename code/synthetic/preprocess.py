from math import floor, pi
import torch as th

def get_data(n, ratio):
    """
    Get synthetic dataset
    :param n: The number of points wanted
    :param ratio: the ratio of test/total desired
    :return train_feats, test_feats, train_labels, test_labels
    """
    # Features
    x = th.rand([n, 2])
    x[:, 0] *= 2*pi
    x[:, 0] -= pi
    x[:, 1] *= 3
    x[:, 1] -= 1.5

    # Labels
    y = (th.sin(x[:, 0] * 2) * 0.5 < x[:, 1]).long()

    return x[floor(n*ratio):, :], x[:floor(n*ratio), :], y[floor(n*ratio):], y[:floor(n*ratio)]