import random

import numpy


def get_array_parties(test_y, result_y):
    deta = (result_y - test_y)
    length = len(deta)
    deta = [deta[i] * deta[i] for i in range(length)]
    parties = 0
    for i in range(length):
        parties += deta[i]
    parties /= length
    return parties


def randx(shape, rand_fun=None, has_constant=True):
    """
        shape: (m, n) m is the number of lines, n is the number of cols
    """
    if not rand_fun:
        # (-1000, 1000)
        rand_fun = lambda x: (x-0.5) * 2000
    n = []
    for i in range(shape[0]):
        line = []
        for j in range(shape[1]):
            if has_constant and j == 0:
                line.append(1)
            else:
                line.append(rand_fun(random.random()))
        n.append(line)

    return numpy.array(n)


def noise(y, noise_fun=None):
    lst_y = y.tolist()
    if not noise_fun:
        n = [(random.random() - 0.5) * 0.2 for i in range(len(lst_y))]
    else:
        n = [noise_fun(random.random()) for i in range(len(lst_y))]
    return numpy.array([lst_y[i] - n[i] for i in range(len(lst_y))])
