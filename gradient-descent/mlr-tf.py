import tensorflow as tf
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略警告


x1 = tf.constant([
137.97,
104.50,
100.00,
124.32,
79.20,
99.00,
124.00,
114.00,
106.69,
138.05,
53.75,
46.91,
68.00,
63.02,
81.26,
86.21
])

x2 = tf.constant([
3,
2,
2,
3,
1,
2,
3,
2,
2,
3,
1,
1,
1,
1,
2,
2
],dtype=tf.float32)

y = tf.constant([
145.00,
110.00,
93.00,
116.00,
65.32,
104.00,
118.00,
91.00,
62.00,
133.00,
51.00,
45.00,
78.50,
69.65,
75.69,
95.30
])

# samples num
n = tf.size(x1)
W = tf.fill([3,1],0.0)
eta = 0.0001
mse = []

min = tf.keras.backend.min(x1)
max = tf.keras.backend.max(x1)
x1 = (x1-min)/(max-min)

one = tf.ones(shape=(n,))
XT = tf.stack([one,x1,x2])
X = tf.linalg.matrix_transpose(XT)
Y = tf.reshape(y,shape=(n,1))

for i in range(0,100000):
    # 2X'(XW-Y)
    dL_dW = 2*tf.linalg.matmul(
        XT,
        (tf.linalg.matmul(X,W)-Y)
    )
    W = W-eta*dL_dW
    loss = tf.math.reduce_sum(tf.linalg.matmul(
        tf.linalg.matrix_transpose(Y-tf.linalg.matmul(X,W)),
        (Y-tf.linalg.matmul(X,W))
    ))
    mse.append(loss)

    if i % 1000 == 0:
        print('loss=',loss.numpy())
print('W: ',W)
plt.figure()
plt.plot(mse)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
