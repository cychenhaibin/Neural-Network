import xmlrpc.client

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 得到数据
iris = datasets.load_iris()
# 特征数据
X = iris.data

# 结果
y = iris.target

# 归一化：把最大和最小都放在0到1里
# print(X.max())
# print(X.min())
X = X/8*0.99+0.01
# print(X)

# 输入x映射到0到1之间的输出值
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 准确率函数
# 计算在y_true和y_predict中相等的元素数量。
# 使用y_true == y_predict这个表达式时，它会返回一个布尔值的数组，其中每个元素表示y_true和y_predict中对应位置的元素是否相等。然后sum函数用于计算这个布尔值数组中值为True的元素数量，也就是相等的元素数量。这个数量表示了正确预测的样本数量。
# 最终，这个值被除以总样本数，从而得到准确率得分（正确预测的样本数量占总样本数的比例）。
def accuasy_score(y_true, y_predict):
    return sum(y_true == y_predict) / len(y_true)

# 训练集，测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 3*3的方阵，对角线为1
oneHot = np.identity(3)
# 处理成0.01到0.99
for i in range(oneHot.shape[0]):
    for j in range(oneHot.shape[1]):
        if(oneHot[i, j] == 1):
            oneHot[i, j] == 0.99
        else:
            oneHot[i, j] == 0.01

y_true = oneHot[y_train]

# 构造W1，W2权重矩阵，随机分布
W1 = np.random.normal(0.0, 1, (4, 8))
W2 = np.random.normal(0.0, 1, (8, 3))
eta = 0.01

# 训练
for i in range(300):
    out1 = np.dot(X_train, W1)
    act1 = sigmoid(out1)      # 隐藏层输出
    out2 = np.dot(act1, W2)
    act2 = sigmoid(out2)
    # 误差
    error = y_true-act2
    hErr = np.dot(error, W2.T)
    # 隐藏层误差
    W2_ = np.dot(act1.T, -error * act2 * (1 - act2))
    W1_ = np.dot(X_train.T, -hErr * act1 * (1 - act1))

    # 更新W1, W2
    W2 -= W2_ * eta
    W1 -= W1_ * eta

# 把想要的结果进行预测
o1 = np.dot(X_test, W1)
a1 = sigmoid(o1)
o2 = np.dot(a1, W2)
a2 = sigmoid(o2)

rs = []
for i in range(a2.shape[0]):
    rs.append(np.argmax(a2[i]))
rs = np.array(rs)
a = accuasy_score(y_test, rs)
print(a)
print(y_test)