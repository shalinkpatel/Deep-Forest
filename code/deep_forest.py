import torch as th
from torch import nn as nn
from torch.functional import split
from deep_tree import Node
from math import floor
from random import shuffle


class DeepForest(nn.Module):
    """
    Deep Forest class. This represents a deep forest, composed of multiple trees.
    """

    def __init__(self, num_trees, depth, num_features, split_ratio, hidden):
        """
        Init function. Initializes all the trees in the forest
        :param num_trees: the number of trees the forest is supposed to have
        :param depth: the depth of the trees
        :param tree_features: lists the features for each tree, as indexes into features
        :param features: the number of features
        :param split_ratio: the ratio of features to be considered.
        :param hidden: the size of the hidden layers for all the trees (same across the whole forest currently)
        """

        self.num_trees = num_trees
        self.tree_features = self.gen_tree_features(num_trees, depth, num_features, split_ratio)

        # trees: a numpy array of all the trees in the forest
        self.trees = []
        for tree_num in range(num_trees):
            tree = Node(self.tree_features[tree_num], hidden, depth)
            self.trees.append(tree)

    def gen_tree_features(num_trees, depth, num_features, split_ratio):
        """
        Function to generate the features subsets for all of the trees
        :param num_trees: the number of trees in the forest
        :param depth: The depth of the each tree
        :param num_features: the number of features in the dataset
        :param split_ratio: the ratio of features to be split on
        """
        tree_features = []
        ctr = 1
        n = floor(split_ratio * num_features)
        for i in range(num_trees):
            feats = {}
            for j in range((depth ** 2) - 1):
                rg = shuffle(list(range(num_features)))
                feats[ctr] = th.tensor(rg[:n])
                ctr += 1
            tree_features.append(feats)
        return tree_features

    def populate_best(self, x, y):
        """
        Precomputation step to find the mode of the left and right split.
        :param x: the input features
        :param y: associated labels
        """
        for tree_num in range(self.num_trees):
            feats = x[self.tree_features[tree_num]]
            self.trees[tree_num].populate_best(feats, y)

    def forward(self, x):
        # TODO: Need to write this
        pass

    def loss(self, x, y):
        """
        Calculate the loss.
        :param x: the input features
        :param y: associated labels
        """
        loss = 0
        for i in range(self.num_trees):
            loss += self.trees[i].loss(x, y)
        return loss

