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

x_train = tf.constant(iris_train[:,:4],tf.float32)
# shape=(120, 5)
X_train = tf.concat([tf.ones([120,1], tf.float32),x_train],axis=1)

# shape=(5, 3)
W = tf.Variable(tf.random.normal([5,3],dtype=tf.float32))

y_train = iris_train[:,-1]
# shape=(120, 3)
Y_train = tf.one_hot(tf.constant(y_train,tf.int32),3)

num_train = len(x_train)

eta = 0.2
for i in range(0,501):
    with tf.GradientTape() as g:
        # ex = tf.exp(X_train@W)
        # Y_p = ex/tf.reduce_sum(ex,1,keepdims=True)
        # equal
        Y_p = tf.nn.softmax(X_train@W)
        Loss = -tf.reduce_sum(Y_p*tf.math.log(Y_p))/num_train

    dL_dW = g.gradient(Loss,W)
    W.assign_sub(eta*dL_dW)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y_p,1),y_train),tf.float32))

    if i % 10 == 0:
        print('i=',i,' loss=',Loss.numpy(), ' accurancy=',acc.numpy())

print(W)
