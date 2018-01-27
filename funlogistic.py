import numpy


class FunLogistic:
    def __init__(self, weight, *args):
        if len(weight.shape) > 1:
            raise Exception('weight of FunLinear should be an (1, n) array, current: ' + str(weight.shape))
        self.weight = weight
        self.args = args

    def get(self, x):
        unit1 = numpy.exp(-self.weight.dot(x))
        return 1/(1 + unit1)

if __name__ == '__main__':
    f = FunLogistic(numpy.array([2, 1]))
    x = numpy.array([[1, 1],
                     [1, 2],
                     [2, 1]])
    y = f.get(numpy.transpose(x))
    print(y)
