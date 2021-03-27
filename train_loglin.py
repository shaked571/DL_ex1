from utils import vocab, L2I, F2I, TRAIN, DEV
import loglinear as ll
import random
import numpy as np
from collections import Counter

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    filtered_feature = Counter([f for f in features if f in vocab])
    vec = np.zeros(len(F2I))
    for k, v in filtered_feature.items():
        vec[F2I[k]] = v
    return vec

def accuracy_on_dataset(dataset, params):
    good, bad = 0, 0
    for label, features in dataset:
        feature_vec = feats_to_vec(features)
        y_tag = ll.predict(feature_vec, params)
        if y_tag == L2I[label]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)

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
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            params[0] -= (learning_rate * grads[0])
            params[1] -= (learning_rate * grads[1])

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(e_i, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    train_data = TRAIN
    dev_data = DEV

    params = ll.create_classifier(len(F2I), len(L2I))
    num_iterations = 100
    learning_rate = 10**-4
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

