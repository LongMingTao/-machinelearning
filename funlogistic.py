import numpy
import matplotlib.pyplot as plot


class FunLogistic:
    def __init__(self, weight, intercept=numpy.array([0]), *args):
        if len(weight.shape) > 1:
            raise Exception('weight of FunLinear should be an (1, n) array, current: ' + str(weight.shape))
        self.weight = weight
        self.intercept = intercept
        self.args = args

    def get(self, x):
        unit1 = numpy.exp(-(self.weight.dot(x) + self.intercept))
        return 1/(1 + unit1)

    def plot_2d(self, plot, r=(-0.5, 0.5), is_show=False):
        x = numpy.arange(r[0], r[1])
        y = - (self.weight[0] * x + self.intercept) / self.weight[1]
        plot.plot(x, y)
        if is_show:
            plot.show()

if __name__ == '__main__':
    f = FunLogistic(numpy.array([100, 2]))
    f2 = FunLogistic(numpy.array([5, 2]))
    x = numpy.array([[1, 1],
                     [1, 2],
                     [2, 1]])
    y = f.get(numpy.transpose(x))
    print(y)