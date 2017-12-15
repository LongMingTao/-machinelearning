from numpy import *
import random


def feature_scaling(train_x, mode=None):
    if not mode or mode == 'mean_normalization':
        for n in range(train_x.shape[1]):
            col_max = train_x[:, n].max()
            col_avg = average(train_x[:, n])
            if col_max == col_avg:
                continue
            for m in range(train_x.shape[0]):
                train_x[m, n] = (train_x[m, n] - col_avg) / col_max
    else:
        pass
    print(train_x)
    return train_x


def gradient_descent(train_x, train_y, alpha):
    train_x = feature_scaling(train_x)
    if train_x.shape[0] != train_y.shape[0]:
        raise Exception('train set err with unequal rank')
    pre_value = zeros((train_x.shape[1], 1))
    cur_value = ones((train_x.shape[1], 1))
    cnt = 0
    while not (cur_value == pre_value).all():
        pre_value = cur_value.copy()
        print('--------------- ' + str(cnt) + ' ---------------')
        cnt += 1
        for col in range(train_x.shape[1]):
            # print('----- ' + str(col) + ' -----')
            # print('pre_value = ' + str(pre_value[col, 0]))
            deta_h = (train_x * pre_value - train_y) * transpose(train_x)[col]
            # print('deta_h = \n' + str(deta_h))
            deta_col = alpha / train_x.shape[0] * deta_h.sum()
            # print('deta_col = ' + str(deta_col))
            cur_value[col, 0] = pre_value[col, 0] + deta_col
            # print('cur_value = ' + str(cur_value[col][0]))
        print('pre_value')
        print(pre_value)
        print('cur_value')
        print(cur_value)


if __name__ == '__main__':
    m = 10
    alpha = -0.1
    train_x = []
    train_y = []
    for n in range(m):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10) * 123
        c = random.random()
        train_x.append((1, a, b, c))
        noice = random.random() + 0.5
        train_y.append((a * n + 2 * b + 3 * c * c) * noice) # 1000 +

    train_x = mat(train_x, dtype=float16)
    train_y = transpose(mat(train_y, dtype=float16))
    # print(train_x)
    # print(train_y)
    power_h = gradient_descent(train_x, train_y, alpha)
