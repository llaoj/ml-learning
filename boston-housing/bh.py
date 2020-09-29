import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略警告

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# 归一化
x_train = (x_train-x_train.min(axis=0))/(x_train.max(axis=0)-x_train.min(axis=0))
n = x_train.shape[0]
x0_train = tf.ones([n,1], tf.float32)
X_train = tf.concat([x0_train,x_train], axis=1)

x_test = (x_test-x_test.min(axis=0))/(x_test.max(axis=0)-x_test.min(axis=0))
n = x_test.shape[0]
x0_test = tf.ones([n, 1], tf.float32)
X_test = tf.concat([x0_test,x_test], axis=1)

Y_train = tf.cast(tf.reshape(y_train,shape=(-1,1)), tf.float32)
Y_test = tf.cast(tf.reshape(y_test,shape=(-1,1)), tf.float32)

mse_train = []
mse_test = []
eta = 0.01
c = x_train.shape[1] + 1
W = tf.Variable(np.random.randn(c,1),dtype=tf.float32)
for i in range(0, 2801):
    # 对W求导
    with tf.GradientTape() as g:
        Loss_train = 0.5*tf.reduce_mean(tf.square(Y_train-tf.matmul(X_train,W)))
    dL_dW = g.gradient(Loss_train,W)

    Loss_test = 0.5*tf.reduce_mean(tf.square(Y_test-tf.matmul(X_test,W)))
    W.assign_sub(eta*dL_dW)

    mse_train.append(Loss_train)
    mse_test.append(Loss_test)

    if i % 100 == 0:
        print("i=",i," Loss_train=",Loss_train.numpy()," Loss_test",Loss_test.numpy())

plt.figure()
plt.plot(mse_train,color='red',linewidth=.5,label='Loss_train')
plt.plot(mse_test,color='blue',linewidth=.5,label='Loss_test')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()