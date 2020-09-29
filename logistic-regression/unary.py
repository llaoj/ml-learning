import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略警告

x = np.array([1,2,3,4])
y = np.array([0,0,1,1])

x = x-x.mean()
print(x)

w = tf.Variable(1.)
b = tf.Variable(1.)

cross_entropy = []
for i in range(0, 101):
    y_p = 1/(1+tf.math.exp(wx+b))
    with tf.GradientTape() as g:
    # broadcast
        loss = -tf.reduce_sum(y*tf.math.log(y_p)+(1-y)*tf.math.log(1-y_p))
    dl_dw,dl_db = g.gradient(loss,[w,b])
    w.assgin_sub(eta*dl_dw)
    b.assgin_sub(eta*dl_db)
    cross_entropy.append(loss)