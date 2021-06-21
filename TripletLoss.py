import torch
import numpy as np


class AnchorPicker():
    def __init__(self, X, Y, output_size: int):
        # X is a torch array, storing samples
        # X[batch][sample][feature]
        self.X = X
        self.Y = Y
        self.Yinv = self.compute_Yinv(self.Y)
        self.output_size = output_size


    def compute_Yinv(self, Y):
        inverse_Y = [{} for i in range(len(Y))]
        for i in range(len(Y)):  # loop through batches
            for current_person in list(set(Y[i])):  # loop through unique persons
                inverse_Y[i][current_person] = [index for index, person in enumerate(Y[i]) if person == current_person]
        return inverse_Y

    def get_anchors(self, batch, model):
        pred_anchors = torch.zeros(len(self.X[batch]), self.output_size) # Initialise
        pred_positives = torch.zeros(len(self.X[batch]), self.output_size)
        pred_negatives = torch.zeros(len(self.X[batch]), self.output_size)

        with torch.no_grad():
            modelX = model(self.X[batch])
        d = distance_matrix(modelX)

        for i, anchor in enumerate(self.X[batch]):
            ID = self.Y[batch][i] # ID of the current person
            pred_anchors[i] = model(anchor)


            positive_pos = hardest_positive_anchor_pos(i, self.Y[batch], self.Yinv[batch], d)
            pos = self.X[batch][positive_pos]
            pred_positives[i] = model(pos)

            negative_pos = hardest_negative_anchor_pos(i, self.Y[batch], self.Yinv[batch], d)
            neg = self.X[batch][negative_pos]
            pred_negatives[i] = model(neg)
        return pred_anchors, pred_positives, pred_negatives


def get_anchors(X_trainbatch, output_size, ):
    pred_anchors = torch.zeros(len(X_train[batch]), output_size) # Initialise
    pred_positives = torch.zeros(len(X_train[batch]), output_size)
    pred_negatives = torch.zeros(len(X_train[batch]), output_size)

    d = distance_matrix(modelX)



def euclidean_distance(X, Y):
    return torch.sum(torch.square(torch.subtract(X, Y)))


def distance_matrix(X):
    d = np.zeros([len(X), len(X)])
    for i in range(0, len(X)):
        for j in range(i+1, len(X)):
            dist = euclidean_distance(X[i], X[j])
            d[i,j] = dist
            d[j,i] = dist
    return d


def hardest_positive_anchor_pos(anchor_position, Y, inverse_Y, d):
    ID = Y[anchor_position]
    sample_positions = inverse_Y[ID]

    current_max_index = 0
    current_max_dist = 0
    for sample_pos in sample_positions:
        if sample_pos != anchor_position:
            dist = d[anchor_position,  sample_pos]
            if dist > current_max_dist:
                current_max_dist = dist
                current_max_index = sample_pos
    return current_max_index


def hardest_negative_anchor_pos(anchor_position, Y, inverse_Y, d):
    # X[anchor_position] = anchor
    ID = Y[anchor_position]
    sample_positions = inverse_Y[ID]

    d_row = np.array(d[anchor_position], copy=True)
    for position in sample_positions:
        d_row[position] = 100
    indices = np.where(d_row == np.amin(d_row))[0]  # returns a tupple, unpack it
    if len(indices) > 1:  # magically, two points are equally close
        return indices[0]
    return indices

# Not used
def normalize_data(modelX):
    def length(a):
        return torch.sqrt(torch.sum(torch.square(a)))
    for i in range(len(modelX)):
        modelX[i] = modelX[i]/length(modelX[i])
