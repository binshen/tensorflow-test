# https://cloud.tencent.com/developer/article/1538680

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
rng = np.random

# 参数
learning_rate = 0.01
training_steps = 1000
display_step = 50

# 训练数据
X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = X.shape[0]

# 随机初始化权重，偏置
W = tf.Variable(rng.randn(),name="weight")
b = tf.Variable(rng.randn(),name="bias")


# 线性回归(Wx+b)
def linear_regression(x):
    return W * x + b


# 均方差
def mean_square(y_pred,y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true,2)) / (2 * n_samples)


# 随机梯度下降优化器
optimizer = tf.optimizers.SGD(learning_rate)


# 优化过程
def run_optimization():
    # 将计算封装在GradientTape中以实现自动微分
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred,Y)

    # 计算梯度
    gradients = g.gradient(loss,[W,b])

    # 按gradients更新 W 和 b
    optimizer.apply_gradients(zip(gradients,[W,b]))


# 针对给定训练步骤数开始训练
for step in range(1,training_steps + 1):
    # 运行优化以更新W和b值
    run_optimization()

    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))


import matplotlib.pyplot as plt

# 绘制图
plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted line')
plt.legend()
plt.show()

