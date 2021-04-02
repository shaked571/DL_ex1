import numpy as np
import random
from utils import create_1_hot_vec, cross_entropy, L2I, softmax, TRAIN, DEV, F2I,TEST, I2L
from train_loglin import feats_to_vec
import pickle

STUDENT = {'name1': 'Refael Shaked Greenfeld',
           'ID1': '305030868',
           'name2': 'Danit Yshaayahu',
           'ID2': '312434269'}


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


def classifier_output(x, params):
    out = x
    for layer in params:
        M, b = layer
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

    all_z = [x]
    all_activation = []
    for (W, b) in params:
        all_z.append(np.array(all_z[-1]).dot(W) + b)
        all_activation.append(np.tanh(all_z[-1]))

    gb = y_tag - y_
    gW = gb * all_activation[-2].reshape(-1, 1)
    grads.append([gW, gb])
    # We start from -2 because for every n layers we have n-1 calculation when we already
    # calculating the first one before the for loop
    for i, layer in enumerate(params[:-1][::-2]):
        cur_ind = len(params) - 2 - i
        gb = (1 - np.power(all_activation[cur_ind], 2)) * grads[-1][1].dot(params[cur_ind+1][0].T)
        gW = gb * all_z[cur_ind].reshape(-1, 1)
        grads.append([gW, gb])

    return loss, grads


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
            for i, (gradW, gradb) in enumerate(grads): # need to go from back to the end. a bit wierd, maybe better the otheer way around...
                params[-1 - i][0] -= (learning_rate * gradW)
                params[-1 - i][1] -= (learning_rate * gradb)

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)

        print(e_i, train_loss, train_accuracy, dev_accuracy)
    return params


def save_parameters(trained_params):
    print("Saving params under the name 'params.pkl'")
    with open("params.pkl", "wb") as f:
        pickle.dump(trained_params, f)


def test_classifier(dataset, params):
    result = []
    for (_, features) in dataset:
        feature_vec = feats_to_vec(features)
        y_tag = predict(feature_vec, params)
        result.append(I2L[y_tag])

    with open('test.pred', 'w') as f:
        f.writelines(result)

def main_test():
    test_data = TEST
    with open('params.pkl', 'rb') as f:
        params = pickle.load(f)
    test_classifier(test_data, params)


def main_train():
    train_data = TRAIN
    dev_data = DEV

    params = create_classifier([len(F2I), 800, 400, 200, 50,  len(L2I)])
    num_iterations = 100
    learning_rate = 10**-4
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    save_parameters(trained_params)


if __name__ == '__main__':
    main_train()
