import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略警告

x = np.array([
137.97,
104.50,
100.00,
126.32,
79.20,
99.00,
124.00,
114.00,
106.69,
140.05,
53.75,
46.91,
68.00,
63.02,
81.26,
86.21
])
y = np.array([
1,
1,
0,
1,
0,
1,
1,
0,
0,
1,
0,
0,
0,
0,
0,
0
])

plt.scatter(x,y)

x = x-np.mean(x)
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())
eta = 0.005

cross_entropy = []
for i in range(0, 6):
    with tf.GradientTape() as g:
        y_p = 1/(1+tf.exp(-(w*x+b)))
        y_p = tf.clip_by_value(y_p,1e-10,1.0)
        loss = -tf.reduce_mean(y*tf.math.log(y_p)+(1-y)*tf.math.log(1-y_p))
    dl_dw,dl_db = g.gradient(loss,[w,b])

    w.assign_sub(eta*dl_dw)
    b.assign_sub(eta*dl_db)
    cross_entropy.append(loss)

    if i % 1 == 0:
        print('i=',i,' loss=',loss)

plt.figure()
plt.scatter(x,y)

plt.show()