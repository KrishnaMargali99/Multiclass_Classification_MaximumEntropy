"""
Goparapu Krishna Margali, kg4060
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lamda = 0.5
alpha = 0.4
epoch = 100
mini_batch_size = 50


def onehotencoding(Y):
    return pd.get_dummies(Y, columns=[4])


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
    return accuracy


def plot_graph():
    plt.plot(losses, color='r', label='Train data_loss')
    plt.plot(losses_test, color='b', label='Test data_loss')
    plt.xlabel('Function epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.savefig()
    plt.show()


def create_mini_batch(X, Y, mini_batch_size):
    batch_data = []
    for i in range(X.shape[0] // mini_batch_size + 1):
        mini_batch_X = X[:, mini_batch_size * i:mini_batch_size * (i + 1)]
        mini_batch_Y = Y[:, mini_batch_size * i:mini_batch_size * (i + 1)]
        batch_data.append((mini_batch_X, mini_batch_Y))
        return batch_data


path = os.getcwd() + '/iris_train.dat'
data = pd.read_csv(path, header=None)

path = os.getcwd() + '/iris_test.dat'
data1 = pd.read_csv(path, header=None)

cols1 = data1.shape[1]
xtest = data1.iloc[:, 0:cols1 - 1]
ytest_in = data1.iloc[:, cols1 - 1:cols1]
ytest = onehotencoding(ytest_in)

pad_ones_test = []
for i in range(xtest.shape[0]):
    pad_ones_test.append(1)
xtest[4] = pad_ones_test
xtest = np.array(xtest.values)
ytest = np.array(ytest.values)
y_actual_test = ytest_in.to_numpy()
W = [[20, 0, 15], [20, 0, 15], [20, 0, 15], [20, 0, 15], [20, 0, 15]]
b = [0.1, 0.1, 0.1]
W = np.array(W)

cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
Y = data.iloc[:, cols - 1:cols]
y = onehotencoding(Y)
pad_ones = []
for i in range(X.shape[0]):
    pad_ones.append(1)
X[4] = pad_ones
y_actual = Y.to_numpy()
X = np.array(X.values)
y = np.array(y.values)
z = np.matmul(X, W) + b
p = softmax(z)

losses = []
losses_test = []
for i in range(epoch):
    mini = create_mini_batch(X, y, mini_batch_size)
    for single in mini:
        xb, yb = single
        z = np.matmul(xb, W) + b
        p = softmax(z)
        ztest = np.matmul(xtest, W) + b
        ptest = softmax(ztest)
        loss = Computeloss(p, W, xb, yb)
        loss_test = Computeloss(ptest, W, xtest, ytest)
        losses.append(loss)
        losses_test.append(loss_test)
        dw = computeGrad(xb, yb, W, p)[0]
        db = computeGrad(xb, yb, W, p)[1]
        W = W - (alpha * dw)
        b = b - (alpha * db)

train_accuracy = check_accuracy(X, W, b, y_actual)
print("The accuracy of the train data : ", train_accuracy * 100, "%")
test_accuracy = check_accuracy(xtest, W, b, y_actual_test)
print("The accuracy of the test data: ", test_accuracy * 100, "%")

plot_graph()
