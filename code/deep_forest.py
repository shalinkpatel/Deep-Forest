import torch as th
from torch import nn as nn
from torch.functional import split
from deep_tree import Node
from math import ceil
from random import shuffle
from math import pi
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import torch.multiprocessing as mp
import os
import time


class DeepForest(nn.Module):
    """
    Deep Forest class. This represents a deep forest, composed of multiple trees.
    """

    def __init__(self, num_trees, depth, num_features, split_ratio, hidden, threaded=True):
        """
        Init function. Initializes all the trees in the forest
        :param num_trees: the number of trees the forest is supposed to have
        :param depth: the depth of the trees
        :param tree_features: lists the features for each tree, as indexes into features
        :param split_ratio: the ratio of features to be considered.
        :param hidden: the size of the hidden layers for all the trees (same across the whole forest)
        :param threaded: whether or not to thread the training process
        """
        super(DeepForest, self).__init__()

        self.threaded = threaded
        if threaded:
            # Number of cores available,
            self.num_processes = os.cpu_count()
            if self.num_processes > num_trees:
                self.num_processes = num_trees

            # Number of trees corrected to be divisible by process count
            offset = num_trees % self.num_processes
            num_trees += self.num_processes - offset

            # Number of trees to handle per process
            self.trees_per_process = int(num_trees / self.num_processes)

        self.num_trees = num_trees

        self.tree_features = self.gen_tree_features(num_trees, depth, num_features, split_ratio)

        self.importance = defaultdict(self.zero)

        self.trees = nn.ModuleList()
        for tree_num in range(num_trees):
            tree = Node(self.tree_features[tree_num], hidden, depth, 1, self.importance)
            self.trees.append(tree)

        # This is required for the ``fork`` method to work
        if threaded:
            self.share_memory()

    def zero(self):
        return 0

    def gen_tree_features(self, num_trees, depth, num_features, split_ratio):
        """
        Function to generate the features subsets for all of the trees
        :param num_trees: the number of trees in the forest
        :param depth: The depth of the each tree
        :param num_features: the number of features in the dataset
        :param split_ratio: the ratio of features to be split on
        """
        tree_features = []
        n = ceil(split_ratio * num_features)
        for i in range(num_trees):
            ctr = 1
            feats = {}
            for j in range((depth ** 2) - 1):
                rg = list(range(num_features))
                shuffle(rg)
                feats[ctr] = th.tensor(rg[:n])
                ctr += 1
            tree_features.append(feats)
        return tree_features

    def populate_best(self, trees, x, y):
        """
        Precomputation step to find the mode of the left and right split.
        :param trees: the slice of trees to populate (nn.moduleList())
        :param x: the input features
        :param y: associated labels
        """
        for tree in trees:
            tree.populate_best(x, y)

    def forward(self, trees, x, device=th.device('cpu')):
        """
        Forward pass function. Calls the forward function of every tree and finds the best
        prediction for every input given all tree predictions
        :param trees: the slice is trees on which to compute (nn.ModuleList())
        :param x: the input features
        :return predictions:
        """
        preds = []
        for tree in trees:
            predictions = tree.forward(x, device)
            preds.append(predictions)
        predictions, _ = th.mode(th.stack(preds, 1), 1)
        return predictions.to(device)

    def loss(self, trees, x, y, device=th.device('cpu')):
        """
        Calculate the loss.
        :param trees: the slice of trees for which to calculate loss (nn.ModuleList())
        :param x: the input features
        :param y: associated labels
        """
        loss = th.tensor([0], dtype=th.float32).to(device)
        for tree in trees:
            loss = tree.loss(x, y, loss, device)
        return loss

    def compute_importance(self, feats):
        """
        Function to tabulate and compute the final importance scores
        :param feats: the background needed for the shapley scores
        """
        for i in trange(self.num_trees):
            self.trees[i].compute_importance(feats)
        total = 0
        for _, v in self.importance.items():
            total += v
        for k, v in self.importance.items():
            self.importance[k] = v / total
        return self.importance

    def train(self, epochs, train_data, train_labels):
        """
        The training function. Calls the appropriate training function depending on whether
        the model was instantiated as threaded or untreaded. The processes are created here
        if threaded.
        :param num_p: the number of the process
        :param trees: the slice of trees for which to calculate loss (nn.ModuleList())
        :param epochs: the number of epochs to run
        :param train_data: the data to train on
        :param train_labels: the labels of the training data
        """
        if self.threaded:
            processes = []
            for num_p in range(self.num_processes):
                # Start routine of process is threaded_train, with a slice of trees
                p = mp.Process(target=self.threaded_train,
                               args=(num_p, self.trees[num_p*self.trees_per_process:(num_p+1)*self.trees_per_process],
                                     epochs, train_data, train_labels))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            self.threaded = False
            self.populate_best(self.trees, train_data, train_labels)
            self.loss(self.trees, train_data, train_labels)
        else:
            self.unthreaded_train(epochs, train_data, train_labels)

    def threaded_train(self, num_p, trees, epochs, train_data, train_labels):
        """
        Training function for a single process. Trains the slice of trees it is given independently of
        other processes.
        :param num_p: the number of the process
        :param trees: the slice of trees for which to calculate loss (nn.ModuleList())
        :param epochs: the number of epochs to run
        :param train_data: the data to train on
        :param train_labels: the labels of the training data
        """
        optimizer = th.optim.Adam(self.parameters())
        for i in range(epochs):
            self.populate_best(trees, train_data, train_labels)
            optimizer.zero_grad()

            loss = self.loss(trees, train_data, train_labels)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print("====THREAD %d====EPOCH %d====\nAcc: %s\nLoss: %s" % (num_p,
                    i, str(th.mean((self.forward(trees, train_data) == train_labels).float())), str(loss)))

    def unthreaded_train(self, epochs, train_data, train_labels):
        """
        Training function without threads. Trains all the trees of the deep forest on a single core.
        :param epochs: the number of epochs to run
        :param train_data: the data to train on
        :param train_labels: the labels of the training data
        """
        optimizer = th.optim.Adam(self.parameters())
        for i in range(epochs):
            self.populate_best(self.trees, train_data, train_labels)
            optimizer.zero_grad()

            loss = self.loss(self.trees, train_data, train_labels)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print("====EPOCH %d====\nAcc: %s\nLoss: %s" % (
                    i, str(th.mean((self.forward(self.trees, train_data) == train_labels).float())), str(loss)))


if __name__ == '__main__':
    # tree: num_trees, depth, num_features, split_ratio, hidden, threaded
    model = DeepForest(10, 3, 2, 1, 10)
    # Unthreaded model
    # model = DeepForest(10, 3, 2, 1, 10, threaded=False)

    # 1000 x 2 ==> batch x features
    x = th.rand([100, 2])
    x[:, 0] *= 2 * pi
    x[:, 0] -= pi
    x[:, 1] *= 3
    x[:, 1] -= 1.5

    # Labels
    y = (th.sin(x[:, 0]) < x[:, 1]).long()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    # Train Epochs, train_data, train_labels
    starttime = time.time()
    model.train(500, x, y)

    print("==============\nFINAL ACC: %s" % str(
            th.mean((model.forward(model.trees, x) == y).float())))
    print("=====training took " + str(time.time() - starttime) + "s=====")

    print(y[:15])
    print(model.forward(model.trees, x, device)[:15].long())
    cdict = {0: 'green', 1: 'purple'}
    plt.scatter(x[:, 0], x[:, 1], c=[cdict[i] for i in model.forward(model.trees, x, device).cpu().numpy()])
    plt.show()

    print(model.compute_importance(x))
