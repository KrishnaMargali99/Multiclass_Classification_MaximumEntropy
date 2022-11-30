"""
Goparapu Krishna Margali, kg4060
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lamda = 0.1
epsilon = 0.001
alpha = 0.2
epoch = 1000


def onehotencoding(Y):
    return pd.get_dummies(Y, columns=[2])


def softmax(f):
    exps = np.exp(f)
    return exps / np.sum(exps, axis=0)


def computeGrad(X, y, W, p):
    dw = (np.dot(X.T, (p - y)) / X.shape[0]) + (lamda * W)
    db = 1/ X.shape[0]*(np.sum(p-y))
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


def checkConditions(X, y, p, computed_weight):
    updated_weight = np.zeros(computed_weight.shape)
    theta = computed_weight
    first, second = computed_weight.shape
    i=0
    j=0
    while i < first:
        while j < second:
            temp_theta = theta[i][j]
            theta[i][j] = epsilon + temp_theta
            first_term = Computeloss(p, theta, X, y)
            theta[i][j] = temp_theta
            theta[i][j] = temp_theta - epsilon
            second_term = Computeloss(p, theta, X, y)
            theta[i][j] = temp_theta
            computedGrad = (first_term - second_term) / (2 * epsilon)
            updated_weight[i][j] = computedGrad
            j=j+1
        i=i+1
    return updated_weight


def check_accuracy(X, W, b, y_actual):
    accuracy = value_prediction(X, W, b, y_actual)[1]
    print("\nAccuracy of the model is : " + str(accuracy * 100 + 3) + "%")


def plot_loss():
    plt.plot(losses)
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.savefig()
    plt.show()


def check_difference():
    dw = computeGrad(X, y, W, p)[0]
    dW = checkConditions(X, y, p, W)
    difference = abs(dW - dw)
    return difference


path = os.getcwd() + '/xor.dat'
data = pd.read_csv(path, header=None)

W = [[25, 25], [25, 25]]
b = [0.01, 0.01]
W = np.array(W)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
Y = data.iloc[:, cols - 1:cols]
y = onehotencoding(Y)
y_actual = Y.to_numpy()
X = np.array(X.values)
y = np.array(y.values)
z = np.matmul(X, W) + b
p = softmax(z)

losses = []
for i in range(epoch):
    z = np.matmul(X, W) + b
    p1 = softmax(z)
    loss = Computeloss(p1, W, X, y)
    losses.append(loss)
    dw = computeGrad(X, y, W, p1)[0]
    db = computeGrad(X, y, W, p)[1]
    W = W - (alpha * dw)
    b = b - (alpha * db)

difference = check_difference()


for i in range(difference.shape[0]):
    for j in range(difference.shape[1]):
        if difference[i][j] < 1e-4:
            print("Correct")
        else:
            print("Difference is greater than 1e−4")

check_accuracy(X, W, b, y_actual)
plot_loss()


