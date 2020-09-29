import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略警告

x = tf.constant([
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

y = tf.constant([
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
],dtype=tf.float32)

x = x-tf.reduce_mean(x)
print(x)

w = tf.Variable(1.)
b = tf.Variable(1.)


eta = 0.005

cross_entropy = []
for i in range(0, 51):
    y_p = 1/(1+tf.exp(-w*x-b))
    with tf.GradientTape() as g:
        loss = -tf.reduce_mean(y*tf.math.log(y_p)+(1-y)*tf.math.log(1-y_p))
    dl_dw,dl_db = g.gradient(loss,[w,b])

    print(dl_dw,dl_db)
    w.assign_sub(eta*dl_dw)
    b.assign_sub(eta*dl_db)
    cross_entropy.append(loss)

    if i % 10 == 0:
        print('i=',i,' loss=',loss)

# plt.figure()
# plt.scatter(x,y)

# plt.show()