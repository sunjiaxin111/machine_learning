import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题


# 线性回归
def linear_regression(alpha=0.01, num_iters=400):
    data = np.loadtxt("data.txt", delimiter=",", dtype=np.float64)
    print("========> Load Data Success!")

    X = data[:, :-1]  # 特征列
    y = data[:, -1].reshape(-1, 1)  # 标签列
    m = len(y)  # 样本数

    # Python参数传递采用的肯定是“传对象引用”的方式。
    # 这种方式相当于传值和传引用的一种综合。
    # 如果函数收到的是一个可变对象（比如字典或者列表）的引用，
    # 就能修改对象的原始值－－相当于通过“传引用”来传递对象。
    # 如果函数收到的是一个不可变对象（比如数字、字符或者元组）的引用，
    # 就不能直接修改原始对象－－相当于通过“传值'来传递对象。
    feature_standardize(X)  # 标准化
    # 画散点图看一下标准化效果
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # numpy的hstack函数为水平地拼接数组，需要保证行数相同
    X = np.hstack((np.ones((m, 1)), X))  # 在X前加一列1
    col = X.shape[1]
    theta = np.zeros((col, 1))

    J_history = gradient_descent(X, y, theta, alpha, num_iters)

    plotJ(J_history, num_iters)


# 标准化
def feature_standardize(X):
    mu = np.mean(X, 0)  # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X, 0)  # 求每一列的标准差
    for i in range(X.shape[1]):  # 遍历列
        X[:, i] = (X[:, i] - mu[i]) / sigma[i]  # 标准化


# 梯度下降算法
def gradient_descent(X, y, theta, alpha, num_iters):
    print("========> Gradient Descent Start!")
    J_history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值

    for i in range(num_iters):  # 遍历迭代次数
        # np.dot(A, B)：对于二维矩阵，计算真正意义上的矩阵乘积，
        # 同线性代数中矩阵乘法的定义。对于一维矩阵，计算两者的内积。
        # 对应元素相乘 element-wise product: np.multiply(), 或 *
        h = np.dot(X, theta)  # 计算样本的预测值
        theta = theta - alpha * (np.dot(np.transpose(X), h - y))  # 梯度下降
        J_history[i] = computer_cost(X, y, theta)  # 调用计算代价函数

    print("========> Gradient Descent End!")
    return J_history


# 计算代价函数
def computer_cost(X, y, theta):
    J = np.dot((np.transpose(np.dot(X, theta) - y)), (np.dot(X, theta) - y)) / 2  # 计算代价J
    return J


# 画每次迭代代价的变化图
def plotJ(J_history, num_iters):
    plt.plot(range(num_iters), J_history)
    plt.xlabel(u"迭代次数", fontproperties=font)  # 注意指定字体，要不然出现乱码问题
    plt.ylabel(u"代价值", fontproperties=font)
    plt.title(u"代价随迭代次数的变化", fontproperties=font)
    plt.show()


if __name__ == "__main__":
    linear_regression(0.0002, 400)
