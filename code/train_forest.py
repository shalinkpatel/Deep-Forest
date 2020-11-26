import torch as th
import numpy as np

def train_model(model, x_train, x_test, y_train, y_test, epochs, log=True, device=th.device('cpu')):
    """
    Main function to train a deep forest model
    :param model: A deep_tree or forest model
    :param x_train: training feats
    :param x_test: testing feats
    :param y_train: training labels
    :param y_test: testing labels
    :param epochs: number of epochs
    :param log: print training
    :param device: a pytorch device for training
    """

    optimizer = th.optim.Adam(model.parameters())
    for i in range(epochs):
        model.populate_best(x_train, y_train)
        optimizer.zero_grad()

        loss = model.loss(x_train, y_train, device)
        loss.backward()
        optimizer.step()

        if i % 200 == 0 and log:
            print("====EPOCH %d====\nAcc: %s\nLoss: %s" % (i, str(th.mean((model.forward(x_test, device) == y_test).float())), str(loss)))

    if log:
        print("==============\nFINAL ACC: %s" % str(th.mean((model.forward(x_test, device) == y_test).float())))
    
    return model, th.mean((model.forward(x_test, device) == y_test).float())