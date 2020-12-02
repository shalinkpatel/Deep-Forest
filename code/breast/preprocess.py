import numpy as np
from sklearn import datasets
import torch as th

def get_data(percent_train):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param percent_train: percentage of data that is used for training (1 - 100)
    :return: Numpy arrays of train data [num_train x (numfeatures = 4)], train labels [num_train], test data
     [num_test x (numfeatures = 4)], test labels [num_test] (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """
    data_dict = datasets.load_breast_cancer()
    data = np.array(data_dict['data'])
    labels = np.array(data_dict['target'])

    np.random.seed(None)
    idx = np.random.permutation(len(data))
    data = data[idx]
    labels = labels[idx]

    num_examples = data.shape[0]
    num_train_examples = int(num_examples*percent_train/100)

    return th.tensor(data[:num_train_examples]).float(), th.tensor(labels[:num_train_examples]).long(), th.tensor(data[num_train_examples:]).float(), th.tensor(labels[num_train_examples:]).long()