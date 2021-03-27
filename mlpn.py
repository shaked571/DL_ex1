import numpy as np
from utils import softmax, cross_entropy, create_1_hot_vec

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


def classifier_output(x, params):
    out = None
    for M, b in zip(params[0::2], params[1::2]):
        if out is None:
            out = np.dot(np.array(x), M) + b
        else:
            out = np.tanh(out)
            out = np.dot(out, M) + b
    return out


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    grads = []
    y_tag = softmax(classifier_output(x, params))
    loss = cross_entropy(y_tag, y)
    y_ = create_1_hot_vec(y, y_tag)

    # last layer
    z = np.array(x).dot(params[0][0]) + params[0][1]
    activation = np.tanh(z)

    gb = y_tag - y_
    gW = gb * activation.reshape(-1, 1)
    grads.append([gW, gb])

    for layer in params[1:]:
        gb = (1 - np.power(activation, 2)) * layer[1].dot(layer[0].T)
        gW = gb * np.array(x).reshape(-1, 1)
        grads.append([gW, gb])
    return None

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for d_in, d_out in zip(dims, dims[1:]):
        M = np.random.randn(d_in, d_out) / np.sqrt(d_in)
        b = np.zeros(d_out)
        layer = [M, b]
        params.append(layer)
    return params

