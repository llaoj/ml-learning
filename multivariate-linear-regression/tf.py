import tensorflow as tf
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

one = tf.ones(shape=(n,))
XT = tf.stack([one,x1,x2])
X = tf.linalg.matrix_transpose(XT)
Y = tf.reshape(y,shape=(n,1))

# W = ((X'X)-1)X'Y
W = tf.linalg.matmul(tf.linalg.matmul(tf.linalg.inv(tf.linalg.matmul(XT,X)),XT),Y)
W = tf.reshape(W,[-1])
yp = W[0] + W[1]*x1 + W[2]*x2

print(W)

fig = plt.figure()
ax =Axes3D(fig)

ax.scatter(x1, x2, y,label='fact')
ax.scatter(x1, x2, yp,color='r',label='predict')

X1,X2 = tf.meshgrid(x1,x2)
YP = W[0] + W[1]*X1 + W[2]*X2
ax.plot_wireframe(X1,X2,YP,color='c',linewidth=.5,label='regression surface')

ax.set_xlabel('Area',color='r')
ax.set_ylabel('Room',color='r')
ax.set_zlabel('Price',color='r')
plt.legend()
plt.show()


