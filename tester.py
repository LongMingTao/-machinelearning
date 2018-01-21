import random

import numpy


def get_array_parties(test_y, result_y):
    deta = (result_y - test_y)
    length = len(deta)
    deta = [abs(deta[i])/test_y[i] for i in range(length)]
    parties = 0
    for i in range(length):
        parties += deta[i]
    parties /= length
    numpy.set_printoptions(threshold='nan')
    print(numpy.column_stack((test_y, result_y, deta)))
    numpy.set_printoptions(threshold=1000)
    return parties


def randx(shape, rand_fun=None, has_constant=True):
    """
        shape: (m, n) m is the number of lines, n is the number of cols
    """
    if not rand_fun:
        # (-100, 100)
        rand_fun = lambda x: (x-0.5) * 200
    n = []
    for i in range(shape[0]):
        line = []
        for j in range(shape[1]):
            if has_constant and j == 0:
                line.append(1)
            else:
                line.append(rand_fun(random.random()))
        n.append(line)

    return numpy.array(n).reshape(shape)


def powerx(x, pow):
    p = [x]
    for i in range(2, pow + 1):
        p.append(x ** i)
    return numpy.column_stack(reversed(p))


def noise(y, loc=0.0, scale=1.0, size=None):
    n = numpy.random.normal(loc, scale, size)
    return y + n
