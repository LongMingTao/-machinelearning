import funlinear as fline
import algorithm as alg
import numpy
import tester
import datatrans as dtra
import matplotlib.pyplot as plt

if __name__ == '__main__':
    m = 100
    t_m = 100
    weight = numpy.array([-5, 2])
    x1 = tester.randx((m, 1), has_constant=False)
    x = tester.powerx(x1, weight.shape[0])

    fun = fline.FunLinear(weight, intercept=numpy.array([25000]))
    y = fun.get(numpy.transpose(x))
    y = tester.noise(y, scale=1000, size=y.shape[0])
    sx, scale, mode = dtra.feature_scaling(x)

    xita, intercept, cnt = alg.gradient_descent(fun, 0.001, sx, y)
    print scale
    print xita
    print intercept
    print cnt

    t_x1 = tester.randx((t_m, 1), has_constant=False)
    t_x = tester.powerx(t_x1, weight.shape[0])
    t_y = fun.get(numpy.transpose(t_x))

    h = fline.FunLinear(xita, intercept=intercept)
    hy = h.get(numpy.transpose(dtra.feature_scaling(t_x, scale)[0]))

    gt = numpy.polyfit(x1.reshape(m), y, weight.shape[0])
    gt_y = numpy.polyval(gt, t_x1)

    plt.scatter(t_x1, gt_y, c='r', marker='o')
    plt.scatter(t_x1, t_y, c='g', marker='>')
    plt.scatter(t_x1, hy, c='b', marker='<')
    plt.show()


