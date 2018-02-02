import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import funlogistic as flogic
import datatrans as dtra
import algorithm as alg
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    X = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                    header=None).values
    X_t = X[::2]
    X_r = X[1::2]
    seto = np.array([(i[0], i[1]) for i in X_t if 'setosa' in i[-1]])
    vers = np.array([(i[0], i[1]) for i in X_t if 'versicolor' in i[-1]])
    virg = np.array([(i[0], i[1]) for i in X_t if 'virginica' in i[-1]])
    fun = flogic.FunLogistic(np.array([0, 0]))
    x_t = np.row_stack((seto, vers, virg))
    sx, scale, mode = dtra.feature_scaling(x_t, mode='sta')
    s_seto = sx[0:25, :]
    s_vers = sx[25:50, :]
    s_virg = sx[50:75, :]
    y = np.array([1 for i in range(25)] + [0 for i in range(50)]).reshape(-1)
    xita, intercept, cnt = alg.gradient_descent(fun, 0.01, sx, y)
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(sx, y)
    xita2 = lr.coef_.reshape(-1)
    fun = flogic.FunLogistic(xita, intercept=intercept)
    fun2 = flogic.FunLogistic(xita2, intercept=lr.intercept_)
    print cnt
    print scale
    print xita
    print xita2

    plt.scatter(s_seto[:, 0], s_seto[:, 1], color='red', marker='o', label='setosa')
    plt.scatter(s_vers[:, 0], s_vers[:, 1], color='blue', marker='x', label='versicolor')
    plt.scatter(s_virg[:, 0], s_virg[:, 1], color='green', marker='+', label='Virginica')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc=2)
    fun.plot_2d(plt, (sx.min(), sx.max()))
    fun2.plot_2d(plt, (sx.min(), sx.max()))
    plt.show()

