import funlogistic as flogic
import algorithm as alg
import numpy
import tester
import datatrans as dtra
import matplotlib.pyplot as plt

if __name__ == '__main__':
    m = 100
    t_m = 100
    weight = numpy.array([5])
    const = numpy.array([1] * m).reshape(m, -1)
    x1 = tester.randx((m, 1), rand_fun=lambda t: (t-0.5) * 20, has_constant=False)
    x = tester.powerx(x1, weight.shape[0])

    fun = flogic.FunLogistic(weight)
    y = fun.get(numpy.transpose(x))
    sx, scale, mode = dtra.feature_scaling(x)

    xita, cnt = alg.gradient_descent(fun, 0.001, sx, y)
    print scale
    print xita
    print cnt

    const2 = numpy.array([1] * t_m).reshape(t_m, -1)
    t_x1 = tester.randx((t_m, 1), rand_fun=lambda t: (t-0.5) * 20, has_constant=False)
    t_x = tester.powerx(t_x1, weight.shape[0])
    t_y = fun.get(numpy.transpose(t_x))

    h = flogic.FunLogistic(xita)
    hy = h.get(numpy.transpose(dtra.feature_scaling(t_x, scale)[0]))

    plt.plot(t_x1, t_y, 'r>')
    plt.plot(t_x1, hy, 'v')
    plt.show()


