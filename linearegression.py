from numpy import *


def normal_equation(train_x, train_y):
    arr_tmp = transpose(train_x).dot(train_x).tolist()
    mat_tmp = mat(arr_tmp).I
    return array(mat_tmp.tolist()).dot(transpose(train_x)).dot(train_y)


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
        return None, cnt
    else:
        return cur_value, cnt
