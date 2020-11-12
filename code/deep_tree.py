import torch as th
from torch import nn as nn

class Leaf(nn.Module):
    """
    Main leaf node class. This is the leaf of a decision tree
    """
    def __init__(self):
        """
        Init function. Does not require any inputs.
        """
        self.__init__()
        self.best = None
    
    def populate_best(self, x, y):
        """
        Determines the best predictor for the node as a constant
        """
        mode, _ = th.mode(y)
        self.best = mode
        return mode

    def forward(self, x):
        # TODO: need to write this
        pass

    def loss(self, x, y):
        # TODO: need to write this
        pass

class Node(nn.Module):
    """
    Main tree class. This represents a deep decision tree with learnable decision boundaries.
    """
    def __init__(self, features, hidden, depth):
        """
        Init function
        - features: a dictionary of which features that a splitter has access to. Represents map depth => tensor of feature index
        - hidden: the hidden size of splitter
        - depth: the depth that the tree has left to construct
        """
        self.__init__()
        self.splitter = nn.Sequential(
            nn.Linear(features[depth].shape[0], hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 2),
            nn.Softmax()
        )

        self.subset = features[depth]
        self.best = []

        if depth == 1:
            self.left = Leaf()
            self.right = Leaf()
        else:
            self.left = Node(features, hidden, depth - 1)
            self.right = Node(features, hidden, depth - 1)

    def populate_best(self, x, y):
        """
        Precomputation step to find the mode of the left and right split.
        - x: the input features
        - y: associated labels
        """
        decision = th.flatten(th.argmax(self.splitter(x[:, subset]), axis=0))
        left_best = th.mode(y[decision == 0])
        right_best = th.mode(y[decision == 1])
        self.best = th.tensor([left_best, right_best])
        self.left.populate_best(x[decision == 0], y[decision == 0])
        self.right.populate_best(x[decision == 1], y[decision == 1])
    
    def forward(self, x):
        # TODO: Need to write this
        pass

    def loss(self, x, y):
        # TODO: Need to write this
        pass