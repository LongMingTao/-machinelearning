from numpy import *
import random


def feature_scaling(train_x, mode=None):
    scaler = []
    work_arr = train_x.copy()
    mode = 'mean_normalization' if not mode else mode
    if mode == 'mean_normalization':
        for n in range(train_x.shape[1]):
            col_s = train_x[:, n].max() - train_x[:, n].min()
            col_avg = average(train_x[:, n])
            scaler.append((col_avg, col_s))
            if col_s == 0:
                continue
            for i in range(work_arr.shape[0]):
                work_arr[i, n] = (work_arr[i, n] - col_avg) / col_s
    else:
        pass
    return work_arr, scaler, mode


def gradient_descent(train_x, train_y, alpha, show_level=1):
    if train_x.shape[0] != train_y.shape[0]:
        raise Exception('train set err with unequal rank')
    pre_value = zeros((train_x.shape[1], 1))
    cur_value = ones((train_x.shape[1], 1))
    cnt = 0
    is_nan = False
    while (not (cur_value == pre_value).all()) and not is_nan:
        pre_value = cur_value.copy()
        for col in range(train_x.shape[1]):
            deta_h = transpose(train_x.dot(pre_value) - train_y).dot(train_x[:, col])
            deta_col = alpha / train_x.shape[0] * deta_h
            cur_value[col, 0] = pre_value[col, 0] + deta_col
            if show_level >= 3:
                print('----- ' + str(col) + ' -----')
                print('pre_value = ' + str(pre_value[col, 0]))
                print('deta_h = \n' + str(deta_h))
                print('deta_col = ' + str(deta_col))
                print('cur_value = ' + str(cur_value[col][0]))
            if 'nan' in str(cur_value[col, 0]) or 'inf' in str(cur_value[col, 0]):
                is_nan = True
                break
        if show_level >= 2:
            print('--------------- ' + str(cnt) + ' ---------------')
            print('pre_value')
            print(pre_value)
            print('cur_value')
            print(cur_value)
        cnt += 1
    if show_level >= 1:
        print('------- final result ------')
        if is_nan:
            print('!!!!!!! nan !!!!!!!')
        print('----- count -----')
        print(cnt)
        print('----- linear regression result -----')
        print(cur_value)
    if is_nan:
        return None
    else:
        return cur_value


def create_train_set(m, n, ground_truth, x_ranges, noise_ranges, dtype=None):
    train_x = []
    train_y = []
    for i in range(m):
        sum = 0
        elements = []
        for j in range(n):
            if list == type(x_ranges[j]) or tuple == type(x_ranges[j]):
                element = random.uniform(x_ranges[j][0], x_ranges[j][1])
            else:
                element = x_ranges[j]
            sum += ground_truth[j] * element
            elements.append(element)
        sum += random.uniform(noise_ranges[0], noise_ranges[1])
        train_x.append(elements)
        train_y.append([sum])
    if dtype is None:
        train_x = array(train_x)
        train_y = array(train_y)
    else:
        train_x = array(train_x, dtype=dtype)
        train_y = array(train_y, dtype=dtype)
    print('----- x, y -----')
    print(concatenate((train_x, train_y), axis=1))
    return train_x, train_y


def test1():
    alpha = -0.01
    train_x = array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
    train_y = array([[6], [12], [18]], dtype=float16)
    gradient_descent(train_x, train_y, alpha, is_fs=True, show_detail=False)
    gradient_descent(train_x, train_y, alpha, is_fs=False, show_detail=False)


def test2():
    m = 10000
    n = 4
    gt = (1000, 1, 6, 8)
    x_range = [1, (-100, 100), (-1000, 1000), (-10, 10)]
    noise_range = [-50, 50]
    alpha = [1.0, 0.1, -1.0, -0.1, -0.01, 0.01]
    train_x, train_y = create_train_set(m, n, gt, x_range, noise_range)
    train_x, scaler, mode = feature_scaling(train_x)
    print('----- scaler for ' + mode + ' -----')
    print(scaler)
    print('----- x, y after scaler -----')
    print(concatenate((train_x, train_y), axis=1))
    for a in alpha:
        print('===== alpha ' + str(a) + ' =====')
        cur_value = gradient_descent(train_x, train_y, a, show_level=0)
        if cur_value is not None:
            print('===== alpha ' + str(a) + ' SUCCESS =====')
            break
        else:
            print('===== alpha ' + str(a) + ' FAIL =====')
    print(cur_value)

if __name__ == '__main__':
    # test1()
    test2()
