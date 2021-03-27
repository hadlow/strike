import numpy as np

def batch_loss(y, a):
    l_sum = np.sum(np.multiply(y, np.log(a)))
    m = y.shape[1]
    l = -(1. / m) * l_sum

    return l

