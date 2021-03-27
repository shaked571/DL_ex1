import numpy as np
from loglinear import softmax
from utils import create_1_hot_vec, cross_entropy

STUDENT = {'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):# W, b, U, b_tag
    W, b, U, b_tag = params
    try:
        out1 = np.dot(np.array(x), W) + b
        activation_1 = np.tanh(out1)
        out2 = np.dot(activation_1, U) +b_tag
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
    y_tag = softmax(classifier_output(x, params))
    loss = cross_entropy(y_tag, y)
    y_ = create_1_hot_vec(y, y_tag)
    gW = (y_tag - y_) * np.array(x).reshape(-1, 1)
    gb = y_tag - y_
    # gU = (y_tag - y_) * np.array(x).reshape(-1, 1)
    # gb_tag = y_tag - y_


    return bla

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
    W = np.zeros((in_dim, hid_dim))
    b = np.zeros(hid_dim)
    U = np.zeros((hid_dim, out_dim))
    b_tag = np.zeros(out_dim)
    return [W, b, U, b_tag]

