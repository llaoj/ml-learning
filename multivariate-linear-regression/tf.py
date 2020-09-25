import tensorflow as tf

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
print(W)
