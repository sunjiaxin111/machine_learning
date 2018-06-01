import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题


# 逻辑回归
def logistic_regression(alpha=0.01, num_iters=400):
    data = np.loadtxt("data2.txt", delimiter=",", dtype=np.float64)
    print("========> Load Data Success!")

    X = data[:, :-1]  # 特征列
    y = data[:, -1].reshape(-1, 1)  # 目标列

    X = map_feature(X[:, 0], X[:, 1])  # 映射为多项式
    col = X.shape[1]
    theta = np.zeros((col, 1))

    theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

    plotJ(J_history, num_iters)

    p = predict(X, theta)  # 预测
    print(u'在训练集上的准确度为%f%%' % np.mean(np.float64(p == y) * 100))  # 与真实值比较，p==y返回True，转化为float


# 映射为多项式
def map_feature(X1, X2):
    degree = 2  # 映射的最高次方
    out = np.ones((X1.shape[0], 1))  # 映射后的结果数组（取代X）
    '''
    这里以degree=2为例，映射为1,x1,x2,x1^2,x1x2,x2^2
    '''
    for i in range(1, degree + 1):  # i取值为1到degree
        for j in range(i + 1):  # j取值为0到i
            # a**b 为a的b次方
            temp = (X1 ** (i - j)) * (X2 ** j)  # * 为对应元素相乘
            out = np.hstack((out, temp.reshape(-1, 1)))
    return out


# sigmoid函数
def sigmoid(z):
    h = 1.0 / (1.0 + np.exp(-z))
    return h


# 梯度下降算法
def gradient_descent(X, y, theta, alpha, num_iters):
    print("========> Gradient Descent Start!")
    J_history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值

    for i in range(num_iters):  # 遍历迭代次数
        h = sigmoid(np.dot(X, theta))  # 计算样本的预测值
        theta = theta - alpha * (np.dot(np.transpose(X), h - y))  # 梯度下降
        J_history[i] = computer_cost(X, y, theta)  # 调用计算代价函数

    print("========> Gradient Descent End!")
    return theta, J_history


# 计算代价函数
def computer_cost(X, y, theta):
    h = sigmoid(np.dot(X, theta))
    # numpy的sum函数，没有axis参数表示全部相加，
    # axis＝0表示按列相加，axis＝1表示按行相加
    # numpy中的log是以e为底的
    J = (-1 / y.shape[0]) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=0)
    return J


# 画每次迭代代价的变化图
def plotJ(J_history, num_iters):
    plt.plot(range(num_iters), J_history)
    plt.xlabel(u"迭代次数", fontproperties=font)  # 注意指定字体，要不然出现乱码问题
    plt.ylabel(u"代价值", fontproperties=font)
    plt.title(u"代价随迭代次数的变化", fontproperties=font)
    plt.show()


# 预测
def predict(X, theta):
    m = X.shape[0]
    p = sigmoid(np.dot(X, theta))  # 预测的结果，是个概率值

    for i in range(m):
        if p[i] > 0.5:  # 概率大于0.5预测为1，否则预测为0
            p[i] = 1
        else:
            p[i] = 0
    return p


if __name__ == "__main__":
    logistic_regression(0.005, 4000)
