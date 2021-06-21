# Transformations

This set of scripts contain all the code used for plots and feature transformations.

## Artifical Neural Networks (ANN)

* ANN_helpers.py contains a set of helper functions to train and run the ANNs such as transforming a biometric database through an ANN and the ANN itself.
* ANN_savemodel.py contains an object to store the neural network. Once training is complete, it will plot the DET curve along other informations for the related train/test combination.
* ANN_to_32b1.py is an example of training of a network, training on vgg1 and testing on vgg2, transforming it to 32 features.
* TripletLoss.py contains the logic for the anchor choices.

## Others
* main.py shows an example of how to convert vgg2 to 32 dimensions using PCA trained on vgg1. It also saves a DET curve for the resulting features.
* DataSerilization.py handles biometric data loading. Note that the biometric data is under NDA.
* Helpers.py handles some miscellaneous operations such as computing a rank-1 identification or saving data to .csv.
* Pca.py contains the PCA class for feature transformations
* plots.py generates all the plots used in the thesis.
* Transforms.py handles data transformations, e.g. for preparing data for ANN training, integer quantisation, binary encoding.

## Folders

* det contains an example of an output DET curve in .pdf, alogn with the points on the graph itself for plotting in another program.
* models contain all the pre-trained triplet-loss models. Format is dimensions_for_batch, where batch is either b1 or b2, e.g. b1 meaning "trained on vgg2, use on vgg1".
