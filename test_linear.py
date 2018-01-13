import funlinear as fline
import algorithm as alg
import numpy
import tester
import datatrans as dtra

if __name__ == '__main__':
    weight = numpy.array([10, -100, 2])
    x = tester.randx((1000, 3))
    fun = fline.FunLinear(weight)
    y = fun.get(numpy.transpose(x))
    sx, scale, mode = dtra.feature_scaling(x)

    xita, cnt = alg.gradient_descent(fline.get_cost_diff, 0.0001, sx, y)
    print scale
    print xita
    print cnt

    t_x = tester.randx((1000, 3))
    t_y = fun.get(numpy.transpose(t_x))

    h = fline.FunLinear(xita)
    hy = h.get(numpy.transpose(dtra.feature_scaling(t_x, scale)[0]))
    print tester.get_array_parties(t_y, hy)



