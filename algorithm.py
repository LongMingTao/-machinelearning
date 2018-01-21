import numpy


def normal_equation(train_x, train_y):
    arr_tmp = numpy.transpose(train_x).dot(train_x).tolist()
    mat_tmp = numpy.mat(arr_tmp).I
    return numpy.array(mat_tmp.tolist()).dot(numpy.transpose(train_x)).dot(train_y)


def get_cost_diff(fun, x, y, weigh=None, punish=0):
    if weigh is not None:
        f = fun.__class__(weigh)
    else:
        f = fun
    sum = []
    sum_unit = f.get(numpy.transpose(x)) - y
    for col in range(x.shape[1]):
        sum.append(sum_unit.dot(x[:, col]))
    incentive = numpy.array(sum).reshape(-1)
    if punish == 0:
        return incentive
    else:
        regular = (fun.weight * fun.weight).reshape(-1) * punish / x.shape[0]
        return incentive + regular


def gradient_descent(fun, alpha, x, y, punish=0, precision=0.001, start_from=None):
    """
    :param fun: cost function
    :param punish: punishment arg of cost diff function
    :param precision: stop when abs(last_result - current_result) < precision
    :param alpha:   learning speed
    :param x:   training feature set. Should be a Numpy array
    :param y:   training result set. Should be a Numpy array
    :param start_from:  optional. Should be a 1*n Numpy array. Given where the algorithm start
    :return:    weigh set. Will be a Numpy array
    """
    if start_from is None:
        start_from = numpy.zeros((x.shape[1]))
    weigh_now = start_from
    weigh_pre = numpy.ones((x.shape[1]))
    over_flow = False
    cnt = 0
    while not (abs(weigh_now - weigh_pre) < precision).all() and not over_flow:
        cnt += 1
        weigh_pre = weigh_now
        weigh_now = weigh_pre - alpha * get_cost_diff(fun, x, y, weigh_pre, punish)
        if 'nan' in str(weigh_now) or 'inf' in str(weigh_now):
            return weigh_now, cnt
    return weigh_now, cnt
