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
        super(DeepForest, self).__init__()

        self.num_trees = num_trees
        self.tree_features = self.gen_tree_features(num_trees, depth, num_features, split_ratio)

        self.importance = defaultdict(lambda: 0)

        # trees: a numpy array of all the trees in the forest
        self.trees = nn.ModuleList()
        for tree_num in range(num_trees):
            tree = Node(self.tree_features[tree_num], hidden, depth, 1, self.importance)
            self.trees.append(tree)

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

    def populate_best(self, x, y):
        """
        Precomputation step to find the mode of the left and right split.
        :param x: the input features
        :param y: associated labels
        """
        for tree_num in range(self.num_trees):
            self.trees[tree_num].populate_best(x, y)

    def forward(self, x, device=th.device('cpu')):
        """
        Forward pass function. Calls the forward function of every tree and finds the best
        prediction for every input given all tree predictions
        :param x: the input features
        :return predictions:
        """
        preds = []
        for tree_num in range(0, self.num_trees):
            predictions = self.trees[tree_num].forward(x, device)
            preds.append(predictions)
        predictions, _ = th.mode(th.stack(preds, 1), 1)
        return predictions.to(device)

    def loss(self, x, y, device=th.device('cpu')):
        """
        Calculate the loss.
        :param x: the input features
        :param y: associated labels
        """
        loss = th.tensor([0], dtype=th.float32).to(device)
        for i in range(self.num_trees):
            loss = self.trees[i].loss(x, y, loss, device)
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
            self.importance[k] = v/total
        return self.importance


if __name__ == '__main__':
    # tree: num_trees, depth, num_features, split_ratio, hidden
    model = DeepForest(10, 3, 2, 1, 10)
    print([p.data for p in model.parameters()] != [])

    # 1000 x 2 ==> batch x features
    x = th.rand([100, 2])
    x[:, 0] *= 2*pi
    x[:, 0] -= pi
    x[:, 1] *= 3
    x[:, 1] -= 1.5

    # Labels
    y = (th.sin(x[:, 0]) < x[:, 1]).long()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    optimizer = th.optim.Adam(model.parameters())
    for i in range(1000):
        model.populate_best(x, y)
        optimizer.zero_grad()

        loss = model.loss(x, y, device)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print("====EPOCH %d====\nAcc: %s\nLoss: %s" % (i, str(th.mean((model.forward(x, device) == y).float())), str(loss)))
    
    print("==============\nFINAL ACC: %s" % str(th.mean((model.forward(x, device) == y).float())))

    print(y[:15])
    print(model.forward(x, device)[:15].long())
    cdict = {0: 'green', 1: 'purple'}
    plt.scatter(x[:, 0], x[:, 1], c=[cdict[i] for i in model.forward(x, device).cpu().numpy()])
    plt.show()

    print(model.compute_importance(x))