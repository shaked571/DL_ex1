import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    x = x - x.max()  # numeric stability
    dominator = np.sum(np.e**x)
    x = np.exp(x) / dominator
    return x
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W, b = params
    try:
        np.dot(np.array(x), W) + b
    except ValueError as e:
        print("Error-W")
        print(W)
        print("Error-b")
        print(b)
    return np.dot(np.array(x), W) + b


def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.

    params: a list of the form [(W, b)]
    W: matrix
    b: vector
    """
    return np.argmax(classifier_output(x, params))


def cross_entropy(y_tag, y):
    """
    We compute only the log of the y_tag in the index that y isn't 0.
    Becuse we sum over all the indices y*np.log(y_tag). and y is 1-hot vector.
    :param y_tag:
    :param y:
    :return:
    """
    return -np.log(y_tag[y])


def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    # we put the softmax here for the loss calc, because in prediction its redundant to calc the prob
    y_tag = softmax(classifier_output(x, params))
    loss = cross_entropy(y_tag, y)
    y_ = create_1_hot_vec(y, y_tag)
    gW = (y_tag - y_) * np.array(x).reshape(-1, 1)
    gb = y_tag - y_
    return loss, [gW, gb]


def create_1_hot_vec(y, y_tag):
    y_ = np.zeros(len(y_tag))
    y_[y] = 1
    return y_


def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W, b]

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print(test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    test4 = softmax(np.array([1001, -1002]))
    print(test4)
    assert np.amax(np.fabs(test4 - np.array([1, 1.282e-800]))) <= 1e-6


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1,2,3], 0, [W,b])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1,2,3], 0, [W,b])

        return loss, grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
