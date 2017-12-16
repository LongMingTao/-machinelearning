from numpy import *


def feature_scaling(train_x, scaler=None, mode=None):
    work_arr = train_x.copy()
    mode = 'mean_normalization' if not mode else mode
    if mode == 'mean_normalization':
        if not scaler:
            scaler = []
            for n in range(train_x.shape[1]):
                col_s = train_x[:, n].max() - train_x[:, n].min()
                col_avg = average(train_x[:, n])
                scaler.append((col_avg, col_s))

        for n in range(work_arr.shape[1]):
            if scaler[n][1] == 0:
                continue
            for i in range(work_arr.shape[0]):
                work_arr[i, n] = (work_arr[i, n] - scaler[n][0]) / scaler[n][1]
    else:
        pass

    return work_arr, scaler, mode
