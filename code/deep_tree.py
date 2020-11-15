import torch as th
from torch import nn as nn


class Leaf(nn.Module):
    """
    Main leaf node class. This is the leaf of a decision tree
    """
    def __init__(self):
        """
        Init function.
        - features: a dictionary of which features that a splitter has access to. Represents map depth => tensor of feature index
        - depth: the depth that the tree has left to construct
        """
        super(Leaf, self).__init__()
        self.best = None

    def populate_best(self, x, y):
        """
        Determines the best predictor for the node as a constant, returns the mode of the inputted labels
        """
        if y.shape[0] != 0:
            mode, _ = th.mode(y)
        else:
            mode = 0
        self.best = mode
        return mode

    def forward(self, x):
        """
        Forward function, returns the last computed best predictor for the leaf
        :param x: inputs to the tree, [num_inputs, num_features]
        :return: the last computed best predictor for the leaf
        """
        return self.best

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
        super(Node, self).__init__()
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
        Pre-computation step to find the mode of the left and right split.
        - x: the input features
        - y: associated labels
        """ 
        if x.shape[0] != 0:
            # apply splitter to get decision boundary
            decision = th.flatten(th.argmax(self.splitter(x[:, self.subset]), axis=1))
            # cases of having no 0 / no 1 / combination of both predictions
            if y[decision == 0].nelement() == 0:
                left_best = 0
                if y[decision == 1].nelement() == 0:
                    right_best = 0
                else:
                    right_best, _ = th.mode(y[decision == 1])
            else:
                left_best, _ = th.mode(y[decision == 0])
                if y[decision == 1].nelement() == 0:
                    right_best = 0
                else:
                    right_best, _ = th.mode(y[decision == 1])
            # put calculated values into self variables
            self.best = th.tensor([left_best, right_best])
            # recursively populate the rest of the tree
            self.left.populate_best(x[decision == 0], y[decision == 0])
            self.right.populate_best(x[decision == 1], y[decision == 1])
        else:
            # if there is no data?
            left_best = 0
            right_best = 0
            self.best = th.tensor([left_best, right_best])
            self.left.populate_best(x, y)
            self.right.populate_best(x, y)
    
    def forward(self, x):
        """
        Forward function, applies the splitter to an input tensor
        :param x: inputs to the tree, [num_inputs, num_features]
        :return: a [num_inputs, 2] tensor representing the split as a probability
        """
        # return the softmax predictions
        return self.splitter(x[:, self.subset])

    def loss(self, x, y):
        # TODO: Need to write this
        pass


if __name__ == '__main__':
    ### System Test for Node

    # 4 x 3 ==> batch x features
    x = th.tensor(
        [
            [1, 2, 3],
            [3, 4, 5],
            [0, -1, 3],
            [6, 5, 4]
        ],
        dtype=th.float32
    )

    # Labels
    y = th.tensor([0, 1, 1, 1])

    # Subset map. Will be randomized when we use the RF
    features = {
        2: th.tensor([0, 2]),
        1: th.tensor([0, 1])
    }
    
    # Construct model
    model = Node(features, 5, 2)
    model.populate_best(x, y)
    print(model.best)

