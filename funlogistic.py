import numpy


class FunLinear:
    def __init__(self, weight, *args):
        if len(weight.shape) > 1:
            raise Exception('weight of FunLinear should be an (1, n) array, current: ' + str(weight.shape))
        self.weight = weight
        self.args = args

    def get(self, x):
        return self.weight.dot(x)

if __name__ == '__main__':
    f = FunLinear(numpy.array([2, 1]))
    x = numpy.array([[1, 1],
                     [1, 2],
                     [2, 1]])
    y = f.get(numpy.transpose(x))
    print(y)
