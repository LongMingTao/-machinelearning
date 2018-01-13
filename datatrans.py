import numpy as np


def feature_scaling(train_x, scale=None, mode=None):
    work_arr = train_x.copy()
    mode = 'mean_normalization' if not mode else mode
    if mode == 'mean_normalization':
        if not scale:
            scale = []
            for n in range(train_x.shape[1]):
                col_s = train_x[:, n].max() - train_x[:, n].min()
                col_avg = np.average(train_x[:, n])
                scale.append((col_avg, col_s))

        for n in range(work_arr.shape[1]):
            if scale[n][1] == 0:
                continue
            for i in range(work_arr.shape[0]):
                work_arr[i, n] = (work_arr[i, n] - scale[n][0]) / scale[n][1]
    else:
        pass

    return work_arr, scale, mode