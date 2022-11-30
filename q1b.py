"""
Goparapu Krishna Margali, kg4060
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lamda = 0.5
alpha = 0.01
epoch = 100


def onehotencoding(Y):
    return pd.get_dummies(Y, columns=[2])


def softmax(f):
    exps = np.exp(f)
    return exps / np.sum(exps, axis=0)


def computeGrad(X, y, W, p):
    dw = (np.dot(X.T, (p - y)) / X.shape[0]) + (lamda * W)
    db = 1 / X.shape[0] * (np.sum(p - y))
    return dw, db


def Computeloss(p, weight, X, y):
    loss = -(np.sum(y * np.log(p))) / X.shape[0] + (np.sum(weight ** 2)) * lamda / 2
    return loss


def value_prediction(X, w, b, y_actual):
    prediction = np.argmax(softmax(np.matmul(X, w) + b), axis=1)
    num = np.sum(y_actual.flatten() == prediction)
    denom = len(prediction)
    accuracy = num / denom
    return prediction, accuracy


def check_accuracy(X, W, b, y_actual):
    accuracy = value_prediction(X, W, b, y_actual)[1]
    print("Accuracy of the model is : " + str(round(accuracy * 100)))


def plot_dataset():
    plt.scatter(X[:100, 0], X[:100, 1], alpha=0.5, color="orange")
    plt.scatter(X[100:200, 0], X[100:200, 1], alpha=0.5, color="black")
    plt.scatter(X[200:, 0], X[200:, 1], alpha=0.5, color="red")
    plt.show()


def setting_points():
    x = np.linspace(-1, 1, 100)
    boundary_1 = np.zeros(100)
    boundary_2 = np.zeros(100)
    boundary_3 = np.zeros(100)
    num = 0
    for i in x:
        boundary_1[num] = (b[0] + W[0][0] * i) / -W[0][1]
        boundary_2[num] = (b[1] + W[1][0] * i) / W[1][1]
        boundary_3[num] = (b[2] + W[2][0] * i) / W[2][1]
        num += 1
    return boundary_1, boundary_2, boundary_3, x


def plot_DecisionBoundaries(x, p1, p2, p3):
    plt.plot(x, p1, color="red")
    plt.plot(x, p2, color="black")
    plt.plot(x, p3, color="orange")
    plt.legend(loc="upper right")
    plot_dataset()
    plt.savefig()
    plt.show()


path = os.getcwd() + '/spiral_train.dat'
data = pd.read_csv(path, header=None)

W = [[25, 25, 25], [25, 25, 25], [25, 25, 25]]
b = [0.1, 0.1, 0.1]
W = np.array(W)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
Y = data.iloc[:, cols - 1:cols]
y = onehotencoding(Y)
pad_ones = []
for i in range(X.shape[0]):
    pad_ones.append(1)
X[2] = pad_ones
y_actual = Y.to_numpy()
X = np.array(X.values)
y = np.array(y.values)
z = np.matmul(X, W) + b
p = softmax(z)

for i in range(epoch):
    z = np.matmul(X, W) + b
    p_softmax = softmax(z)
    dw = computeGrad(X, y, W, p_softmax)[0]
    db = computeGrad(X, y, W, p)[1]
    W = W - (alpha * dw)
    b = b - (alpha * db)

check_accuracy(X, W, b, y_actual)
boundary_1, boundary_2, boundary_3, x = setting_points()
plot_DecisionBoundaries(x, boundary_1, boundary_2, boundary_3)
