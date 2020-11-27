import numpy as np
from sklearn import datasets
import torch as th


def get_data(pct_train):
    """
    :param pct_train: what percentage of the data should be used to train on
    :return: as numpy arrays: train_data, train_labels, test_data, test_labels
    """
    data_dict = datasets.load_wine()
    data = np.array(data_dict['data'])
    labels = np.array(data_dict['target'])

    np.random.seed(None)
    idx = np.random.permutation(len(data))
    data = data[idx]
    labels = labels[idx]

    num_examples = data.shape[0]
    num_train_examples = int(num_examples*pct_train/100)

    train_data = th.tensor(data[:num_train_examples]).float()
    train_labels = th.tensor(labels[:num_train_examples])
    test_data = th.tensor(data[num_train_examples:]).float()
    test_labels = th.tensor(labels[num_train_examples:])

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = get_wine_data(80)
    print("done loading")

