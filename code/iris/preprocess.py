import torch as th
import numpy as np
from sklearn import datasets

def get_data(percent_train):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param percent_train: percentage of data that is used for training (1 - 100)
    :return: Numpy arrays of train data [num_train x (numfeatures = 4)], train labels [num_train], test data
     [num_test x (numfeatures = 4)], test labels [num_test] (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """ 
    np.random.seed(None)
    # Switch to ratio
    ratio = percent_train / 100
    # Load the dataset as np_arrays
    iris = datasets.load_iris()
    data = np.array(iris['data'])
    labels = np.array(iris['target'])
    # Shuffle the data and labels
    idx = np.random.permutation(len(data))
    data = data[idx]
    labels = labels[idx]
    num_examples = len(labels)
    num_train = int(num_examples * ratio)
    # Slice relative to the given ratio
    train_data, train_labels = th.tensor(data[:num_train]).float(), th.tensor(labels[:num_train])
    test_data, test_labels = th.tensor(data[num_train:]).float(), th.tensor(labels[num_train:])
    # Return tuple of data and labels
    return train_data, train_labels, test_data, test_labels