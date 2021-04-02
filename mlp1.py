import numpy as np
import random
from utils import create_1_hot_vec, cross_entropy, L2I, F2I, TRAIN, DEV, softmax
from train_loglin import feats_to_vec

STUDENT = {'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


def accuracy_on_dataset(dataset, params):
    good, bad = 0, 0
    for label, features in dataset:
        feature_vec = feats_to_vec(features)
        y_tag = predict(feature_vec, params)
        if y_tag == L2I[label]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def classifier_output(x, params): # W, b, U, b_tag
    W, b, U, b_tag = params
    try:
        out1 = np.dot(np.array(x), W) + b
        activation_1 = np.tanh(out1)
        out2 = np.dot(activation_1, U) + b_tag
    except ValueError as e:
        print("Error-W")
        print(W)
        print("Error-b")
        print(b)
        print("Error-U")
        print(U)
        print("Error-b_tag")
        print(b_tag)
        exit(1)
        return
    return out2

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    y_tag = softmax(classifier_output(x, params))

    loss = cross_entropy(y_tag, y)
    y_ = create_1_hot_vec(y, y_tag)

    z = np.array(x).dot(W) + b
    activation = np.tanh(z)

    gb_tag = y_tag - y_
    gU = gb_tag * activation.reshape(-1, 1)

    gb = (1 - np.power(activation, 2)) * gb_tag.dot(U.T)
    gW = gb * np.array(x).reshape(-1, 1)

    return loss, [gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    # W = np.zeros((in_dim, hid_dim))
    # b = np.zeros(hid_dim)
    # U = np.zeros((hid_dim, out_dim))
    # b_tag = np.zeros(out_dim)
    W = np.random.randn(in_dim, hid_dim) / np.sqrt(in_dim)
    b = np.zeros(hid_dim)

    U = np.random.randn(hid_dim, out_dim) / np.sqrt(hid_dim)
    b_tag = np.zeros(out_dim)
    return [W, b, U, b_tag]

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for e_i in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:

            x = feats_to_vec(features)   # convert features to a vector.
            y = L2I[label]                    # convert the label to number if needed.
            loss, grads = loss_and_gradients(x, y, params)
            cum_loss += loss
            for i, grad in enumerate(grads):
                params[i] -= (learning_rate * grad)

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(e_i, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    # for xor
    # from xor_data import data
    # train_data = data
    # dev_data = data

    train_data = TRAIN
    dev_data = DEV

    params = create_classifier(len(F2I), 1000, len(L2I))
    num_iterations = 100
    learning_rate = 10**-4
    # xor learning rate
    # learning_rate = 10**-1
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

