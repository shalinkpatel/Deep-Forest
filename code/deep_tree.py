import torch as th
from torch import nn as nn
import matplotlib.pyplot as plt
from math import pi


class Leaf(nn.Module):
    """
    Main leaf node class. This is the leaf of a decision tree
    """
    def __init__(self):
        """
        Init function.
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
        y = th.tensor([self.best], dtype=th.float32)
        return y.repeat_interleave(x.shape[0])


    def loss(self, x, y, loss):
        return loss


class Node(nn.Module):
    """
    Main tree class. This represents a deep decision tree with learnable decision boundaries.
    """
    def __init__(self, features, hidden, depth, id):
        """
        Init function
        - features: a dictionary of which features that a splitter has access to. Represents map id => tensor of feature index
        - hidden: the hidden size of splitter
        - depth: the depth that the tree has left to construct
        """
        super(Node, self).__init__()
        self.splitter = nn.Sequential(
            nn.Linear(features[depth].shape[0], hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
            nn.Softmax()
        )

        self.subset = features[id]
        self.best = []

        if depth == 1:
            self.left = Leaf()
            self.right = Leaf()
        else:
            id += 1
            self.left = Node(features, hidden, depth - 1, id)
            id += 1
            self.right = Node(features, hidden, depth - 1, id)

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
            self.best = th.tensor([left_best, right_best], dtype=th.long)
            # recursively populate the rest of the tree
            self.left.populate_best(x[decision == 0], y[decision == 0])
            self.right.populate_best(x[decision == 1], y[decision == 1])
        else:
            # if there is no data?
            left_best = 0
            right_best = 0
            self.best = th.tensor([left_best, right_best], dtype=th.long)
            self.left.populate_best(x, y)
            self.right.populate_best(x, y)
    
    def forward(self, x):
        """
        Forward function, applies the splitter to an input tensor recursively (cascades data through tree)
        :param x: inputs to the tree, [num_inputs, num_features]
        :return: the predictions for the inputs [num_inputs]
        """
        # return the softmax predictions
        splits = self.splitter(x[:, self.subset])
        left_indices = splits[:, 0] >= 0.5
        right_indices = splits[:, 0] < 0.5
        left_data = x[left_indices]
        right_data = x[right_indices]
        y_pred = th.zeros_like(splits[:, 0])

        y_pred[left_indices] = self.left.forward(left_data)
        y_pred[right_indices] = self.right.forward(right_data)

        return y_pred

    def loss(self, x, y, loss):
        """
        Loss function, applies the backpropagation of the loss recursively through the tree
        :param x: inputs to the tree, [num_inputs, num_features]
        :param y: associated labels
        :param loss: loss value, inputted at initial value of 0
        :return: total loss
        """
        # Get the left and right split
        split = self.splitter(x[:, self.subset])
        left = split[:, 0]
        right = split[:, 1]
        # Get the label one-hot vecor
        y_hot = nn.functional.one_hot(y, num_classes=-1)
        # Left and right weight for cross-entropy
        left_weighted = y_hot * left[:, None]
        right_weighted = y_hot * right[:, None]

        left_best = self.best[0].repeat(x.shape[0])
        right_best = self.best[1].repeat(x.shape[0])
        
        loss += nn.functional.cross_entropy(left_weighted, left_best.type(th.LongTensor))
        loss += nn.functional.cross_entropy(right_weighted, right_best.type(th.LongTensor))
        loss = self.left.loss(x, y, loss)
        loss = self.right.loss(x, y, loss)
        return loss


if __name__ == '__main__':
    ### System Test for Node

    # 1000 x 2 ==> batch x features
    x = th.rand([10000, 2])
    x[:, 0] *= 2*pi
    x[:, 0] -= pi
    x[:, 1] *= 3
    x[:, 1] -= 1.5

    # Labels
    y = (th.sin(x[:, 0]) < x[:, 1]).long()

    # Subset map. Will be randomized when we use the RF
    features = {
        7: th.tensor([0, 1]),
        6: th.tensor([0, 1]),
        5: th.tensor([0, 1]),
        4: th.tensor([0, 1]),
        3: th.tensor([0, 1]),
        2: th.tensor([0, 1]),
        1: th.tensor([0, 1])
    }
    
    # Construct model
    model = Node(features, 10, 3, 1)
    print(model.best)

    print([p.data for p in model.parameters()])

    # Train
    optimizer = th.optim.Adam(model.parameters())
    for i in range(1000):
        model.populate_best(x, y)
        optimizer.zero_grad()

        loss = model.loss(x, y, th.tensor([0], dtype=th.float32))
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("====EPOCH %d====\nAcc: %s\nLoss: %s" % (i, str(th.mean((model.forward(x) == y).float())), str(loss)))
    
    print("==============\nFINAL ACC: %s" % str(th.mean((model.forward(x) == y).float())))

    print(y[:15])
    print(model.forward(x)[:15].long())
    cdict = {0: 'green', 1: 'purple'}
    plt.scatter(x[:, 0], x[:, 1], c=[cdict[i] for i in model.forward(x).numpy()])
    plt.show()
