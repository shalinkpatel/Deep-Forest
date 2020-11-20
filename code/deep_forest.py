import torch as th
from torch import nn as nn
from deep_tree import Node


class DeepForest(nn.Module):
    """
    Deep Forest class. This represents a deep forest, composed of multiple trees.
    """

    def __init__(self, num_trees, depth, tree_features, features, hidden):
        """
        Init function. Initializes all the trees in the forest
        :param num_trees: the number of trees the forest is supposed to have
        :param depth: the depth of the trees
        :param tree_features: lists the features for each tree, as indexes into features
        :param features: the features themselves
        :param hidden: the size of the hidden layers for all the trees (same across the whole forest currently)
        """

        self.num_trees = num_trees
        self.tree_features = tree_features

        # trees: a numpy array of all the trees in the forest
        self.trees = []
        for tree_num in range(num_trees):
            tree = Node(features[tree_features[tree_num]], hidden, depth)
            self.trees.append(tree)
        self.trees = np.array(self.trees)

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
        # TODO: Need to write this
        pass

