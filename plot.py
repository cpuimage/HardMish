# !/usr/bin/env python
# coding: utf-8

from numpy import exp, log, tanh, linspace, clip
import seaborn as sns
import matplotlib.pyplot as plt


def mish(x):
    act = x * tanh(log(1 + exp(x)))
    return act


def relu(x):
    return max(0, x)


def sigmoid(x):
    sg = 1 / (1 + exp(-x))
    return sg


def swish(x):
    return x * sigmoid(x)


def hard_mish(x):
    return clip(x + 2., 0., 2.) * 0.5 * x


def main():
    x = linspace(-3, 3, 10000)
    y = mish(x)
    y1 = sigmoid(x)
    y2 = [relu(i) for i in x]
    y4 = tanh(x)
    y5 = swish(x)
    y6 = hard_mish(x)
    plt.figure(1, figsize=(20, 10))
    plt.subplot(231)
    sns.lineplot(x=x, y=y1, color='red', label='Sigmoid Activation')
    plt.title('Sigmoid Activation')
    plt.subplot(232)
    sns.lineplot(x=x, y=y, color='blue', label='Mish Activation')
    plt.title('Mish Activation')
    plt.subplot(233)
    sns.lineplot(x=x, y=y2, color='green', label='ReLU Activation')
    plt.title('ReLU Activation')
    plt.subplot(234)
    sns.lineplot(x=x, y=y4, color='green', label='tanh Activation')
    plt.title('tanh Activation')
    plt.subplot(236)
    sns.lineplot(x=x, y=y5, color='red', label='Swish Activation')
    plt.title('Swish Activation')
    plt.subplot(235)
    sns.lineplot(x=x, y=y6, color='red', label='Hard Mish Activation')
    plt.title('Hard Mish Activation')
    plt.savefig('plot.png')
    plt.show()


if __name__ == '__main__':
    main()
