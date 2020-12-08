# deep-learning-final-project

This repository contains the code for the Deep Forest classification model. This model was inspired by the Random Forest algorithm. Here, data splitting to the left and right child of each node in the decision tree is determined by an MLP of three layers.

To instantiate our model, use the following call:

DeepForest(num_trees, depth, num_features, split_ratio, hidden, threaded=True)
:param num_trees: the number of trees the forest is supposed to have
:param depth: the depth of the trees
:param tree_features: lists the features for each tree, as indexes into features
:param split_ratio: the ratio of features to be considered.
:param hidden: the size of the hidden layers for all the trees (same across the whole forest)
:param threaded: whether or not to thread the training process, default is True

To train the model, use the following call:

model.train(epochs, train_data, train_labels)
:param epochs: the number of epochs to run
:param train_data: the data to train on [num_inputs, num_features]
:param train_labels: the labels of the training data [num_inputs]

Our model was trained on a synthetic dataset (classifying whether a point was above or below a sine curve) and on three UCI datasets (iris, wine and breast cancer datasets) and compared with a Random Forest classifier (from sci-kit learn) and a standard three-layer MLP. These testing and comparison jupyter notebook files can be found under code/<dataset>/benchmark.ipynb, where <dataset> is `synthetic`, `iris`, `wine` or `breast`.

Here are the necessary dependencies to run our model:
-torch
-shap
-sklearn
-seaborn
-matplotlib
-numpy

# Folders - Here is a description of our code folder, containing our model

`code`: the code folder contains deep_forest.py (the deep forest class, the class of the final model), deep_tree.py (the decision tree file, that contains node and leaf classes), and the dataset folders (Iris, Wine, Synthetic and Breast). Here is a description of these files.

	
`deep_forest.py`: this file contains the code the for the DeepForest class, the class of the model. To instantiate this model, the arguments are as follows: 

DeepForest(num_trees, depth, num_features, split_ratio, hidden, threaded=True)
:param num_trees: the number of trees the forest is supposed to have
:param depth: the depth of the trees
:param tree_features: lists the features for each tree, as indexes into features
:param split_ratio: the ratio of features to be considered.
:param hidden: the size of the hidden layers for all the trees (same across the whole forest)
:param threaded: whether or not to thread the training process, default is True

This class contains the training function for the model, whose arguments are the number of epochs to train, the train data, and the train labels. The call is made as follows:
model.train(epochs, train_data, train_labels)
:param epochs: the number of epochs to run
:param train_data: the data to train on [num_inputs, num_features]
:param train_labels: the labels of the training data [num_inputs]


`deep_tree.py`: this file contains the code to build the decision tree and run the pre-computation, forward pass and loss calculation of the tree to train the model. The importance of the features at each node is also calculated here for the tree for the interpretation aspect of our model.


`code/<dataset>`:


`synthetic`: this folder contains the data for the synthetic dataset of our testing suite, which was classifying whether a point was above or below a sine curve. This folder contains the pre-processing file (preprocess.py) where the data is created, and the jupypter notebook file that trains our model on it and compares it to our benchmark models (Random Forest and standard three-layer MLP).


`iris`: this folder contains the data for the iris UCI dataset of our testing suite. This folder contains the pre-processing file (preprocess.py) where the data is created, and the jupypter notebook file that trains our model on it and compares it to our benchmark models (Random Forest and standard three-layer MLP).

`wine`: this folder contains the data for the wine UCI dataset of our testing suite. This folder contains the pre-processing file (preprocess.py) where the data is created, and the jupypter notebook file that trains our model on it and compares it to our benchmark models (Random Forest and standard three-layer MLP).

`breast`: this folder contains the data for the breast UCI dataset of our testing suite. This folder contains the pre-processing file (preprocess.py) where the data is created, and the jupypter notebook file that trains our model on it and compares it to our benchmark models (Random Forest and standard three-layer MLP).
