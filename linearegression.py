from numpy import *
import random


def feature_scaling(train_x, key=None, mode=None):
    if 'array' not in str(type(key)):
        work_arr = train_x.copy()
    else:
        if key.shape[1] == train_x.shape[1]:
            work_arr = key.copy()
        else:
            raise Exception('key array has unequal column with train_x')
    if not mode or mode == 'mean_normalization':
        for n in range(train_x.shape[1]):
            col_s = train_x[:, n].max() - train_x[:, n].min()
            col_avg = average(train_x[:, n])
            if col_s == 0:
                continue
            for i in range(work_arr.shape[0]):
                work_arr[i, n] = (work_arr[i, n] - col_avg) / col_s
    else:
        pass
    return work_arr


def gradient_descent(train_x, train_y, alpha, is_fs=True, show_detail=False):
    if is_fs:
        pre_x = train_x
        train_x = feature_scaling(train_x)
    else:
        pass
    if train_x.shape[0] != train_y.shape[0]:
        raise Exception('train set err with unequal rank')
    pre_value = zeros((train_x.shape[1], 1))
    cur_value = ones((train_x.shape[1], 1))
    cnt = 0
    is_nan = False
    while (not (cur_value == pre_value).all()) and not is_nan:
        pre_value = cur_value.copy()
        for col in range(train_x.shape[1]):
            # print('----- ' + str(col) + ' -----')
            # print('pre_value = ' + str(pre_value[col, 0]))
            deta_h = transpose(train_x.dot(pre_value) - train_y).dot(train_x[:, col])
            # print('deta_h = \n' + str(deta_h))
            deta_col = alpha / train_x.shape[0] * deta_h
            # print('deta_col = ' + str(deta_col))
            cur_value[col, 0] = pre_value[col, 0] + deta_col
            # print('cur_value = ' + str(cur_value[col][0]))
            if 'nan' in str(cur_value[col, 0]) or 'inf' in str(cur_value[col, 0]):
                is_nan = True
                break
        if show_detail:
            print('--------------- ' + str(cnt) + ' ---------------')
            print('pre_value')
            print(pre_value)
            print('cur_value')
            print(cur_value)
        cnt += 1
    print('------- final result ------')
    if is_nan:
        print('!!!!!!! nan !!!!!!!')
    print('----- count -----')
    print(cnt)
    print('----- linear regression result -----')
    if is_fs:
        print(feature_scaling(pre_x, transpose(cur_value)))
    else:
        print(cur_value)
    if is_nan:
        return False
    else:
        return True


def test1():
    alpha = -0.01
    train_x = array([[1, 1, 2], [1, 3, 4], [1, 5, 6]], dtype=float16)
    train_y = array([[6], [12], [18]], dtype=float16)
    gradient_descent(train_x, train_y, alpha, is_fs=True, show_detail=False)
    gradient_descent(train_x, train_y, alpha, is_fs=False, show_detail=False)


def test2():
    m = 10000
    alpha = [1.0, 0.1, -1.0, -0.1, -0.01, 0.01]
    train_x = []
    train_y = []
    for n in range(m):
        a = random.randint(-100, 100)
        b = random.randint(-10, 10) * 123
        c = random.randint(-10, 10)
        train_x.append([1, a, b, c])
        noice = (random.random() - 0.5) * 50
        train_y.append([(1000 + a + 6 * b + 8 * c) + noice])
    train_x = array(train_x, dtype=float64)
    train_y = array(train_y, dtype=float64)
    print('----- x, y -----')
    print(concatenate((train_x, train_y), axis=1))
    for a in alpha:
        print('===== alpha ' + str(a) + ' =====')
        if gradient_descent(train_x, train_y, a):
            print('===== alpha ' + str(a) + ' SUCCESS =====')
            break
        else:
            print('===== alpha ' + str(a) + ' FAIL =====')
            # gradient_descent(train_x, train_y, alpha * 0.1, False)


if __name__ == '__main__':
    # test1()
    test2()
