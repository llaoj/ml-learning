import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略警告

URL_TRAIN = 'http://download.tensorflow.org/data/iris_training.csv'
path_train = tf.keras.utils.get_file(
    URL_TRAIN.split('/')[-1],
    URL_TRAIN)
iris_train = np.array(pd.read_csv(path_train,header=0))

# URL_TEST = 'http://download.tensorflow.org/data/iris_test.csv'
# path_test = tf.keras.utils.get_file(
#     URL_TEST.split('/')[-1],
#     URL_TEST)
# iris_test = np.array(pd.read_csv(path_test,header=0))

x_train = iris_train[:,0:2]
y_train = iris_train[:,-1]
x_train = x_train[y_train<2]
x_train = x_train-x_train.mean(axis=0)
y_train = y_train[y_train<2]
# plt.scatter(x_train[:,0],x_train[:,1],c=y_train.reshape(-1,1))

num=len(x_train)
x0 = np.ones(num).reshape(-1,1)
X_train = tf.cast(tf.concat([x0,x_train],axis=1),dtype=tf.float32)
Y_train = tf.cast(y_train.reshape(-1,1),dtype=tf.float32)

W = tf.Variable(np.random.randn(3,1),dtype=tf.float32)
eta = 0.02
cross_entropy = []
accuracy = []

# x1 = [-1.5,1.5]
# x2 = (W[0]+W[1]*x1)/W[2]
# plt.plot(x1,x2)
# plt.xlim([-1.5,1.5])
# plt.ylim([-1.5,1.5])

for i in range(0, 1001):
    with tf.GradientTape() as g:
        pred = 1/(1+tf.exp(-tf.matmul(X_train,W)))
        Loss_train = -tf.reduce_mean(Y_train*tf.math.log(pred)+(1-Y_train)*tf.math.log(1-pred))
    dL_dW = g.gradient(Loss_train,W)
    W.assign_sub(eta*dL_dW)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred >.5,1.,0.),Y_train),tf.float32))
    cross_entropy.append(Loss_train)
    accuracy.append(acc)

    # x2 = (W[0]+W[1]*x1)/W[2]
    # plt.plot(x1,x2)

    if i % 10 == 0:
        print('i=',i,' Loss_train=',Loss_train.numpy(),' Accuracy=',acc.numpy())



plt.plot(cross_entropy,label='cross_entropy_loss')
plt.plot(accuracy,label='accuracy')

plt.legend()
plt.show()