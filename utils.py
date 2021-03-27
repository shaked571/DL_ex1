# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import os
def read_data(fname):
    data = []
    with open(fname) as f:
        for line in f.readlines():
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data

root_data_path = os.path.join(os.path.dirname(os.path.abspath((__file__))), 'data')
train_p = os.path.join(root_data_path, 'train')
dev_p = os.path.join(root_data_path, 'dev')
def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text,text[1:])]

TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data(train_p)]
DEV = [(l,text_to_bigrams(t)) for l,t in read_data(dev_p)]

from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x,c in fc.most_common(600)])

# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

