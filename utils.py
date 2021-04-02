import os
from collections import Counter
import numpy as np

STUDENT = {'name1': 'Refael Shaked Greenfeld',
           'ID1': '305030868',
           'name2': 'Danit Yshaayahu',
           'ID2': '312434269'}


def read_data(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    x = x - x.max()  # numeric stability
    dominator = np.sum(np.e ** x)
    x = np.exp(x) / dominator
    return x


root_data_path = os.path.join(os.path.dirname(os.path.abspath((__file__))), 'data')
train_p = os.path.join(root_data_path, 'train')
dev_p = os.path.join(root_data_path, 'dev')
test_p = os.path.join(root_data_path, 'test')


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def text_to_unigrams(text):
    unigrams = []
    not_a_letter = {"!", ",", "(", ")", "[", "]", ".", "?", "<", ">", "*", "&", "^", "%", "$", "#", "@", "~", "+", "=",
                    "{", "}", "\\", "|", "'", ";", "`", " ", "\n", "\t", "\r", '"', ":", "/", "_", "-"}
    for c in text:
        if c not in not_a_letter and c not in list(range(10)):
            unigrams.append(c)
    return unigrams


def create_data_set(path, text_to_features):
    return [(label, text_to_features(t)) for label, t in read_data(path)]


# unigrams
# TRAIN = create_data_set(train_p, text_to_unigrams)
# DEV = create_data_set(dev_p, text_to_unigrams)
# TEST = create_data_set(test_p, text_to_unigrams)

# bigrams
TRAIN = create_data_set(train_p, text_to_bigrams)
DEV = create_data_set(dev_p, text_to_bigrams)
TEST = create_data_set(test_p, text_to_bigrams)

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
I2L = {l: i for i, l in L2I.items()}
# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}


def cross_entropy(y_tag, y):
    """
    We compute only the log of the y_tag in the index that y isn't 0.
    Becuse we sum over all the indices y*np.log(y_tag). and y is 1-hot vector.
    :param y_tag:
    :param y:
    :return:
    """
    return -np.log(y_tag[y])


def create_1_hot_vec(y, y_tag):
    y_ = np.zeros(len(y_tag))
    y_[y] = 1
    return y_
