import funlinear as fline
import algorithm as alg
import numpy
import tester
import datatrans as dtra
import matplotlib.pyplot as plt

if __name__ == '__main__':
    m = 100
    t_m = 100
    weight = numpy.array([-5, 2, 3])
    const = numpy.array([1] * m).reshape(m, -1)
    x1 = tester.randx((m, 1), has_constant=False)
    x = numpy.column_stack((tester.powerx(x1, weight.shape[0] - 1), const))

    fun = fline.FunLinear(weight)
    y = fun.get(numpy.transpose(x))
    y = tester.noise(y, scale=1000, size=y.shape[0])
    sx, scale, mode = dtra.feature_scaling(x)

    xita, cnt = alg.gradient_descent(fun, 0.00001, sx, y)
    print scale
    print xita
    print cnt

    const2 = numpy.array([1] * t_m).reshape(t_m, -1)
    t_x1 = tester.randx((t_m, 1), has_constant=False)
    t_x = numpy.column_stack((tester.powerx(t_x1, weight.shape[0] - 1), const2))
    t_y = fun.get(numpy.transpose(t_x))

    h = fline.FunLinear(xita)
    hy = h.get(numpy.transpose(dtra.feature_scaling(t_x, scale)[0]))

    gt = numpy.polyfit(x1.reshape(m), y, weight.shape[0] - 1)
    gt_y = numpy.polyval(gt, t_x1)

    plt.plot(t_x1, gt_y, 'g<')
    plt.plot(t_x1, t_y, 'r>')
    plt.plot(t_x1, hy, 'v')
    plt.show()


